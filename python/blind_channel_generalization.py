
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = list(range(2001, 2021))
SNR_VALUES = [0, 5, 10, 15, 20]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
NUM_TAPS = 8
CP_LENGTH = 16

PILOT_SPACING = 4
PILOTS = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA = np.setdiff1d(np.arange(NUM_SUBCARRIERS), PILOTS)

CALIBRATION_FRAMES = 20

SWITCH_1 = 300
SWITCH_2 = 600

EWMA_ALPHA = 0.90
WINDOW_LENGTH = 20
CONFIRMATION_FRAMES = 5

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    "Stationary Exponential",
    "Stationary Uniform",
    "Stationary Two-Cluster",
    "Dynamic E-U-E",
    "Dynamic U-E-U"
]

# Normalized channel power delay profiles.
PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS),

    # Unseen profile: two separated multipath clusters.
    "Two-Cluster": np.array([
        0.45, 0.35, 0.0, 0.0,
        0.15, 0.05, 0.0, 0.0
    ], dtype=float)
}

for name in PROFILES:
    PROFILES[name] /= np.sum(PROFILES[name])


# ============================================================
# 2. QPSK
# ============================================================

def qpsk_modulate(bits):

    pairs = bits.reshape(-1, 2)

    return (
        (1 - 2 * pairs[:, 0])
        + 1j * (1 - 2 * pairs[:, 1])
    ) / np.sqrt(2)


def qpsk_demodulate(symbols):

    bits = np.empty((len(symbols), 2), dtype=np.int8)

    bits[:, 0] = (symbols.real < 0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0).astype(np.int8)

    return bits.reshape(-1)


# ============================================================
# 3. CHANNEL COVARIANCE MODELS
# ============================================================

def calculate_covariance(profile):

    indices = np.arange(NUM_SUBCARRIERS)

    difference = (
        indices[:, None] - indices[None, :]
    )

    covariance = np.zeros(
        (NUM_SUBCARRIERS, NUM_SUBCARRIERS),
        dtype=complex
    )

    for tap, power in enumerate(profile):

        covariance += power * np.exp(
            -2j * np.pi * difference * tap
            / NUM_SUBCARRIERS
        )

    return covariance


COVARIANCES = {
    name: calculate_covariance(PROFILES[name])
    for name in ["Exponential", "Uniform"]
}


def prepare_models(noise_variance):

    models = {}

    for name, covariance in COVARIANCES.items():

        pilot_covariance = covariance[
            np.ix_(PILOTS, PILOTS)
        ]

        cross_covariance = covariance[:, PILOTS]

        matrix = (
            pilot_covariance
            + noise_variance * np.eye(len(PILOTS))
        )

        inverse_matrix = np.linalg.inv(matrix)

        sign, logdet = np.linalg.slogdet(matrix)

        if sign <= 0:
            raise ValueError("Invalid covariance matrix")

        models[name] = {
            "estimator": cross_covariance @ inverse_matrix,
            "inverse": inverse_matrix,
            "logdet": float(logdet)
        }

    return models


def estimate_channel(pilot_values, model):

    return model["estimator"] @ pilot_values


def calculate_nll(pilot_values, model):

    solution = model["inverse"] @ pilot_values

    quadratic = np.real(
        np.vdot(pilot_values, solution)
    )

    return model["logdet"] + quadratic


def ls_estimate(pilot_values):

    x = np.arange(NUM_SUBCARRIERS)

    return (
        np.interp(x, PILOTS, pilot_values.real)
        + 1j * np.interp(x, PILOTS, pilot_values.imag)
    )


# ============================================================
# 4. SCENARIO DEFINITIONS
# ============================================================

def actual_profile(scenario, frame):

    if scenario == "Stationary Exponential":
        return "Exponential"

    if scenario == "Stationary Uniform":
        return "Uniform"

    if scenario == "Stationary Two-Cluster":
        return "Two-Cluster"

    if scenario == "Dynamic E-U-E":

        if frame < SWITCH_1:
            return "Exponential"

        if frame < SWITCH_2:
            return "Uniform"

        return "Exponential"

    if scenario == "Dynamic U-E-U":

        if frame < SWITCH_1:
            return "Uniform"

        if frame < SWITCH_2:
            return "Exponential"

        return "Uniform"

    raise ValueError(f"Unknown scenario: {scenario}")


def phase_name(scenario, frame):

    if scenario.startswith("Stationary"):
        return "Stationary"

    if frame < SWITCH_1:
        return "Phase 1"

    if frame < SWITCH_2:
        return "Phase 2"

    return "Phase 3"


def expected_transitions(scenario):

    if scenario == "Dynamic E-U-E":

        return [
            (SWITCH_1 + 1, "Exponential", "Uniform"),
            (SWITCH_2 + 1, "Uniform", "Exponential")
        ]

    if scenario == "Dynamic U-E-U":

        return [
            (SWITCH_1 + 1, "Uniform", "Exponential"),
            (SWITCH_2 + 1, "Exponential", "Uniform")
        ]

    return []


# ============================================================
# 5. GENERATE ONE OFDM FRAME
# ============================================================

def generate_frame(rng, profile_name, noise_variance):

    bits = rng.integers(
        0, 2,
        len(DATA) * 2,
        dtype=np.int8
    )

    symbols = qpsk_modulate(bits)

    frequency_symbol = np.zeros(
        NUM_SUBCARRIERS,
        dtype=complex
    )

    frequency_symbol[PILOTS] = 1.0 + 0j
    frequency_symbol[DATA] = symbols

    time_symbol = (
        np.fft.ifft(frequency_symbol)
        * np.sqrt(NUM_SUBCARRIERS)
    )

    transmitted = np.concatenate([
        time_symbol[-CP_LENGTH:],
        time_symbol
    ])

    powers = PROFILES[profile_name]

    taps = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    taps *= np.sqrt(powers)

    true_channel = np.fft.fft(
        taps,
        NUM_SUBCARRIERS
    )

    received = np.convolve(transmitted, taps)

    noise = np.sqrt(noise_variance / 2) * (
        rng.normal(size=len(received))
        + 1j * rng.normal(size=len(received))
    )

    received += noise

    received_no_cp = received[
        CP_LENGTH:
        CP_LENGTH + NUM_SUBCARRIERS
    ]

    received_frequency = (
        np.fft.fft(received_no_cp)
        / np.sqrt(NUM_SUBCARRIERS)
    )

    return {
        "bits": bits,
        "pilot_values": received_frequency[PILOTS],
        "received_data": received_frequency[DATA],
        "true_channel": true_channel
    }


# ============================================================
# 6. EWMA DETECTOR
# ============================================================

class EWMADetector:

    def __init__(self, initial_profile):

        self.selected = initial_profile
        self.score = 0.0
        self.initialized = False

        self.candidate = None
        self.count = 0

    def update(self, difference):

        if not self.initialized:

            self.score = difference
            self.initialized = True

        else:

            self.score = (
                EWMA_ALPHA * self.score
                + (1 - EWMA_ALPHA) * difference
            )

        proposed = (
            "Uniform"
            if self.score > 0
            else "Exponential"
        )

        old_profile = self.selected
        switched = False

        if proposed == self.selected:

            self.candidate = None
            self.count = 0

        else:

            if proposed == self.candidate:
                self.count += 1

            else:
                self.candidate = proposed
                self.count = 1

            if self.count >= CONFIRMATION_FRAMES:

                self.selected = proposed
                switched = True

                self.candidate = None
                self.count = 0

                self.score = 0.0
                self.initialized = False

        return self.selected, switched, old_profile


# ============================================================
# 7. SLIDING-WINDOW DETECTOR
# ============================================================

class WindowDetector:

    def __init__(self, initial_profile):

        self.selected = initial_profile
        self.window = []

        self.candidate = None
        self.count = 0

    def update(self, difference):

        self.window.append(difference)

        if len(self.window) > WINDOW_LENGTH:
            self.window.pop(0)

        old_profile = self.selected
        switched = False

        if len(self.window) < WINDOW_LENGTH:

            return self.selected, switched, old_profile

        mean_difference = float(np.mean(self.window))

        proposed = (
            "Uniform"
            if mean_difference > 0
            else "Exponential"
        )

        if proposed == self.selected:

            self.candidate = None
            self.count = 0

        else:

            if proposed == self.candidate:
                self.count += 1

            else:
                self.candidate = proposed
                self.count = 1

            if self.count >= CONFIRMATION_FRAMES:

                self.selected = proposed
                switched = True

                self.candidate = None
                self.count = 0

                self.window.clear()

        return self.selected, switched, old_profile


# ============================================================
# 8. BLIND INITIALIZATION
# ============================================================

def blind_initialization(calibration_observations, models):

    total_exp = 0.0
    total_uni = 0.0

    for pilot_values in calibration_observations:

        total_exp += calculate_nll(
            pilot_values,
            models["Exponential"]
        )

        total_uni += calculate_nll(
            pilot_values,
            models["Uniform"]
        )

    selected = (
        "Uniform"
        if total_uni < total_exp
        else "Exponential"
    )

    return selected, total_exp, total_uni


# ============================================================
# 9. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(seed, snr_db, scenario):

    rng = np.random.default_rng(seed)

    noise_variance = 10 ** (-snr_db / 10)

    models = prepare_models(noise_variance)

    # --------------------------------------------------------
    # Blind calibration: no profile label is given to receiver.
    # --------------------------------------------------------

    calibration_observations = []

    for frame in range(CALIBRATION_FRAMES):

        profile = actual_profile(scenario, frame)

        generated = generate_frame(
            rng,
            profile,
            noise_variance
        )

        calibration_observations.append(
            generated["pilot_values"]
        )

    initial_selection, initial_nll_exp, initial_nll_uni = (
        blind_initialization(
            calibration_observations,
            models
        )
    )

    detectors = {
        "EWMA": EWMADetector(initial_selection),
        "Sliding Window": WindowDetector(initial_selection)
    }

    methods = [
        "Perfect CSI",
        "LS",
        "Fixed Exponential LMMSE",
        "Fixed Uniform LMMSE",
        "EWMA Adaptive LMMSE",
        "Window Adaptive LMMSE"
    ]

    if scenario.startswith("Dynamic"):
        phases = ["Phase 1", "Phase 2", "Phase 3"]
    else:
        phases = ["Stationary"]

    errors = {
        method: {phase: 0 for phase in phases}
        for method in methods
    }

    nmse_sum = {
        method: {phase: 0.0 for phase in phases}
        for method in methods
        if method != "Perfect CSI"
    }

    total_bits = {phase: 0 for phase in phases}
    frame_counts = {phase: 0 for phase in phases}

    selection_correct = {
        detector: {phase: 0 for phase in phases}
        for detector in detectors
    }

    histories = {
        detector: []
        for detector in detectors
    }

    switch_events = {
        detector: []
        for detector in detectors
    }

    # --------------------------------------------------------
    # Process remaining frames after blind calibration.
    # --------------------------------------------------------

    for frame in range(CALIBRATION_FRAMES, NUM_FRAMES):

        true_profile_name = actual_profile(
            scenario,
            frame
        )

        phase = phase_name(scenario, frame)

        generated = generate_frame(
            rng,
            true_profile_name,
            noise_variance
        )

        bits = generated["bits"]
        pilot_values = generated["pilot_values"]
        received_data = generated["received_data"]
        true_channel = generated["true_channel"]

        # ----------------------------------------------------
        # Baseline estimates
        # ----------------------------------------------------

        ls = ls_estimate(pilot_values)

        fixed_exp = estimate_channel(
            pilot_values,
            models["Exponential"]
        )

        fixed_uni = estimate_channel(
            pilot_values,
            models["Uniform"]
        )

        estimates = {
            "Perfect CSI": true_channel,
            "LS": ls,
            "Fixed Exponential LMMSE": fixed_exp,
            "Fixed Uniform LMMSE": fixed_uni
        }

        # ----------------------------------------------------
        # Adaptive estimates
        # ----------------------------------------------------

        for detector_name, detector in detectors.items():

            selected = detector.selected

            adaptive_estimate = estimate_channel(
                pilot_values,
                models[selected]
            )

            if detector_name == "EWMA":
                method = "EWMA Adaptive LMMSE"
            else:
                method = "Window Adaptive LMMSE"

            estimates[method] = adaptive_estimate

            # Profile accuracy is only defined when the true
            # profile is one of the two candidate profiles.
            if true_profile_name in ["Exponential", "Uniform"]:

                if selected == true_profile_name:
                    selection_correct[detector_name][phase] += 1

            histories[detector_name].append({
                "Frame": frame + 1,
                "Actual Profile": true_profile_name,
                "Selected Profile": selected,
                "Phase": phase
            })

        # ----------------------------------------------------
        # BER and NMSE
        # ----------------------------------------------------

        true_power = np.mean(
            np.abs(true_channel) ** 2
        )

        for method, estimate in estimates.items():

            channel_data = estimate[DATA]

            equalized = (
                received_data * np.conj(channel_data)
                / (np.abs(channel_data) ** 2 + 1e-12)
            )

            detected_bits = qpsk_demodulate(equalized)

            errors[method][phase] += np.count_nonzero(
                detected_bits != bits
            )

            if method != "Perfect CSI":

                frame_nmse = (
                    np.mean(
                        np.abs(estimate - true_channel) ** 2
                    )
                    / max(true_power, 1e-12)
                )

                nmse_sum[method][phase] += frame_nmse

        total_bits[phase] += len(bits)
        frame_counts[phase] += 1

        # ----------------------------------------------------
        # Update detector after current frame.
        # ----------------------------------------------------

        nll_exp = calculate_nll(
            pilot_values,
            models["Exponential"]
        )

        nll_uni = calculate_nll(
            pilot_values,
            models["Uniform"]
        )

        difference = nll_exp - nll_uni

        for detector_name, detector in detectors.items():

            new_selection, switched, old_profile = (
                detector.update(difference)
            )

            if switched:

                switch_events[detector_name].append({
                    "Decision Frame": frame + 1,
                    "Effective Frame": frame + 2,
                    "Old Profile": old_profile,
                    "New Profile": new_selection
                })

    # ========================================================
    # 10. METRIC TABLE
    # ========================================================

    result_rows = []

    for phase in phases:

        for method in methods:

            result_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Scenario": scenario,
                "Phase": phase,
                "Method": method,
                "BER": (
                    errors[method][phase]
                    / total_bits[phase]
                ),
                "NMSE": (
                    np.nan
                    if method == "Perfect CSI"
                    else nmse_sum[method][phase]
                    / frame_counts[phase]
                ),
                "Selection Accuracy": np.nan
            })

        for detector_name in detectors:

            accuracy_defined = (
                scenario != "Stationary Two-Cluster"
            )

            accuracy = (
                selection_correct[detector_name][phase]
                / frame_counts[phase]
                if accuracy_defined
                else np.nan
            )

            result_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Scenario": scenario,
                "Phase": phase,
                "Method": f"{detector_name} Selection Accuracy",
                "BER": np.nan,
                "NMSE": np.nan,
                "Selection Accuracy": accuracy
            })

    # ========================================================
    # 11. SWITCH AND DETECTION METRICS
    # ========================================================

    event_rows = []

    for detector_name, events in switch_events.items():

        # A switch is counted as false if the new model is
        # incorrect when it first becomes active.
        false_events = []

        for event in events:

            effective_frame = event["Effective Frame"]

            true_profile_at_use = actual_profile(
                scenario,
                effective_frame - 1
            )

            if event["New Profile"] != true_profile_at_use:

                false_events.append(event)

        # Detection delays for known two-profile transitions.
        transitions = expected_transitions(scenario)

        history = histories[detector_name]

        for switch_frame, old_profile, new_profile in transitions:

            correct_frames = [
                record["Frame"]
                for record in history
                if (
                    record["Frame"] >= switch_frame
                    and record["Selected Profile"] == new_profile
                )
            ]

            if correct_frames:

                detection_frame = min(correct_frames)

                detection_delay = (
                    detection_frame - switch_frame
                )

            else:

                detection_frame = -1
                detection_delay = -1

            event_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Scenario": scenario,
                "Detector": detector_name,
                "Event": f"{old_profile} to {new_profile}",
                "Detection Frame": detection_frame,
                "Detection Delay": detection_delay,
                "False Switches": np.nan
            })

        event_rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Scenario": scenario,
            "Detector": detector_name,
            "Event": "Total Incorrect Switches",
            "Detection Frame": np.nan,
            "Detection Delay": np.nan,
            "False Switches": len(false_events)
        })

    # ========================================================
    # 12. HISTORY TABLE
    # ========================================================

    history_rows = []

    for detector_name, records in histories.items():

        for record in records:

            history_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Scenario": scenario,
                "Detector": detector_name,
                **record
            })

    calibration_row = {
        "Seed": seed,
        "SNR (dB)": snr_db,
        "Scenario": scenario,
        "True Initial Profile": actual_profile(scenario, 0),
        "Selected Initial Profile": initial_selection,
        "Initial Exponential NLL": initial_nll_exp,
        "Initial Uniform NLL": initial_nll_uni,
        "Initial Selection Correct": (
            initial_selection == actual_profile(scenario, 0)
            if scenario != "Stationary Two-Cluster"
            else np.nan
        )
    }

    return (
        result_rows,
        event_rows,
        history_rows,
        calibration_row
    )


# ============================================================
# 13. MAIN EXPERIMENT
# ============================================================

def main():

    all_results = []
    all_events = []
    all_history = []
    all_calibration = []

    total_runs = (
        len(SEEDS)
        * len(SNR_VALUES)
        * len(SCENARIOS)
    )

    completed = 0

    print("\nBLIND INITIALIZATION AND GENERALIZATION")
    print("---------------------------------------")
    print("Seeds:", len(SEEDS))
    print("SNR values:", SNR_VALUES)
    print("Scenarios:", SCENARIOS)
    print("Calibration frames:", CALIBRATION_FRAMES)
    print("Total experiments:", total_runs)

    for scenario in SCENARIOS:

        for snr_db in SNR_VALUES:

            for seed in SEEDS:

                (
                    results,
                    events,
                    history,
                    calibration
                ) = run_experiment(
                    seed,
                    snr_db,
                    scenario
                )

                all_results.extend(results)
                all_events.extend(events)
                all_history.extend(history)
                all_calibration.append(calibration)

                completed += 1

                print(
                    f"Progress: {completed}/{total_runs} | "
                    f"{scenario} | "
                    f"SNR {snr_db} dB | "
                    f"Seed {seed}"
                )

    raw = pd.DataFrame(all_results)
    event_df = pd.DataFrame(all_events)
    history_df = pd.DataFrame(all_history)
    calibration_df = pd.DataFrame(all_calibration)

    # ========================================================
    # SAVE RAW FILES
    # ========================================================

    raw.to_csv(
        OUTPUT_DIR / "blind_generalization_raw.csv",
        index=False
    )

    event_df.to_csv(
        OUTPUT_DIR / "blind_generalization_events.csv",
        index=False
    )

    history_df.to_csv(
        OUTPUT_DIR / "blind_generalization_history.csv",
        index=False
    )

    calibration_df.to_csv(
        OUTPUT_DIR / "blind_initialization_results.csv",
        index=False
    )

    # ========================================================
    # BLIND INITIALIZATION SUMMARY
    # ========================================================

    calibration_summary = (
        calibration_df[
            calibration_df["Scenario"] != "Stationary Two-Cluster"
        ]
        .groupby(["Scenario", "SNR (dB)"])
        .agg(
            Initial_Accuracy=("Initial Selection Correct", "mean")
        )
        .reset_index()
    )

    calibration_summary.to_csv(
        OUTPUT_DIR / "blind_initialization_summary.csv",
        index=False
    )

    # ========================================================
    # BER AND NMSE SUMMARY
    # ========================================================

    metric_methods = [
        "Perfect CSI",
        "LS",
        "Fixed Exponential LMMSE",
        "Fixed Uniform LMMSE",
        "EWMA Adaptive LMMSE",
        "Window Adaptive LMMSE"
    ]

    metric_df = raw[
        raw["Method"].isin(metric_methods)
    ]

    summary = (
        metric_df.groupby([
            "Scenario",
            "SNR (dB)",
            "Phase",
            "Method"
        ])
        .agg(
            Mean_BER=("BER", "mean"),
            Std_BER=("BER", "std"),
            Mean_NMSE=("NMSE", "mean"),
            Std_NMSE=("NMSE", "std")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "blind_generalization_summary.csv",
        index=False
    )

    # ========================================================
    # SELECTION ACCURACY SUMMARY
    # ========================================================

    accuracy_df = raw[
        raw["Method"].str.contains(
            "Selection Accuracy",
            na=False
        )
    ]

    accuracy_summary = (
        accuracy_df.groupby([
            "Scenario",
            "SNR (dB)",
            "Phase",
            "Method"
        ])
        .agg(
            Mean_Accuracy=("Selection Accuracy", "mean"),
            Std_Accuracy=("Selection Accuracy", "std")
        )
        .reset_index()
    )

    accuracy_summary.to_csv(
        OUTPUT_DIR / "blind_selection_accuracy_summary.csv",
        index=False
    )

    # ========================================================
    # FALSE SWITCH SUMMARY
    # ========================================================

    false_df = event_df[
        event_df["False Switches"].notna()
    ]

    false_summary = (
        false_df.groupby([
            "Scenario",
            "SNR (dB)",
            "Detector"
        ])
        .agg(
            Mean_Incorrect_Switches=("False Switches", "mean"),
            Std_Incorrect_Switches=("False Switches", "std")
        )
        .reset_index()
    )

    false_summary.to_csv(
        OUTPUT_DIR / "blind_false_switch_summary.csv",
        index=False
    )

    # ========================================================
    # DETECTION DELAY SUMMARY
    # ========================================================

    delay_df = event_df[
        event_df["Detection Delay"].notna()
    ]

    if not delay_df.empty:

        delay_summary = (
            delay_df.groupby([
                "Scenario",
                "SNR (dB)",
                "Detector",
                "Event"
            ])
            .agg(
                Mean_Delay=("Detection Delay", "mean"),
                Detection_Rate=(
                    "Detection Delay",
                    lambda values: np.mean(values >= 0)
                )
            )
            .reset_index()
        )

    else:

        delay_summary = pd.DataFrame()

    delay_summary.to_csv(
        OUTPUT_DIR / "blind_detection_delay_summary.csv",
        index=False
    )

    # ========================================================
    # BER GRAPHS
    # ========================================================

    for scenario in SCENARIOS:

        scenario_phases = (
            ["Phase 1", "Phase 2", "Phase 3"]
            if scenario.startswith("Dynamic")
            else ["Stationary"]
        )

        for phase in scenario_phases:

            plt.figure(figsize=(10, 6))

            subset = summary[
                (summary["Scenario"] == scenario)
                & (summary["Phase"] == phase)
            ]

            for method in metric_methods:

                data = subset[
                    subset["Method"] == method
                ].sort_values("SNR (dB)")

                plt.plot(
                    data["SNR (dB)"],
                    data["Mean_BER"],
                    marker="o",
                    label=method
                )

            plt.xlabel("SNR (dB)")
            plt.ylabel("Mean BER")
            plt.title(
                f"Blind Channel Generalization - {scenario} - {phase}"
            )

            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            filename = (
                scenario.lower().replace(" ", "_")
                + "_"
                + phase.lower().replace(" ", "_")
                + "_ber.png"
            )

            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=300
            )

            plt.close()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n========================================")
    print("BLIND GENERALIZATION COMPLETED")
    print("========================================")

    print("\nINITIAL BLIND IDENTIFICATION ACCURACY:")
    print(calibration_summary.to_string(index=False))

    print("\nFALSE / INCORRECT SWITCH SUMMARY:")
    print(false_summary.to_string(index=False))

    print("\nDETECTION DELAY SUMMARY:")
    print(delay_summary.to_string(index=False))

    print("\nTWO-CLUSTER GENERALIZATION RESULTS:")

    ood_summary = summary[
        summary["Scenario"] == "Stationary Two-Cluster"
    ]

    print(
        ood_summary[
            [
                "SNR (dB)",
                "Method",
                "Mean_BER",
                "Mean_NMSE"
            ]
        ].to_string(index=False)
    )

    print("\nDYNAMIC ADAPTIVE BER RESULTS:")

    dynamic_summary = summary[
        summary["Scenario"].isin([
            "Dynamic E-U-E",
            "Dynamic U-E-U"
        ])
        & summary["Method"].isin([
            "EWMA Adaptive LMMSE",
            "Window Adaptive LMMSE"
        ])
    ]

    print(
        dynamic_summary[
            [
                "Scenario",
                "SNR (dB)",
                "Phase",
                "Method",
                "Mean_BER",
                "Mean_NMSE"
            ]
        ].to_string(index=False)
    )

    print("\nAll results saved in results/")

    print("\nIMPORTANT INTERPRETATION:")
    print("- Initial model selection uses pilot observations only.")
    print("- Two-Cluster is outside the candidate model set.")
    print("- Two-Cluster selection accuracy is not reported.")
    print("- Calibration frames are excluded from post-calibration BER.")
    print("- Results are simulation evidence, not real radio measurements.")


if __name__ == "__main__":
    main()
