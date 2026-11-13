
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = list(range(1001, 1021))  # 20 independent seeds

SNR_VALUES = [0, 5, 10, 15, 20]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
NUM_TAPS = 8
CP_LENGTH = 16

PILOT_SPACING = 4
PILOTS = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA = np.setdiff1d(np.arange(NUM_SUBCARRIERS), PILOTS)

SWITCH_1 = 300
SWITCH_2 = 600

EWMA_ALPHA = 0.90
WINDOW_LENGTH = 20
CONFIRMATION_FRAMES = 5

T_CRITICAL = 2.093  # Approx. 95% two-sided t critical value, df=19

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    "Stationary Exponential",
    "Stationary Uniform",
    "Dynamic E-U-E"
]

PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for profile_name in PROFILES:
    PROFILES[profile_name] /= np.sum(PROFILES[profile_name])


# ============================================================
# 2. QPSK MODULATION AND DEMODULATION
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
# 3. CHANNEL COVARIANCE MATRICES
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
    name: calculate_covariance(profile)
    for name, profile in PROFILES.items()
}


# ============================================================
# 4. CHANNEL PROFILE SELECTION
# ============================================================

def actual_profile(scenario, frame):

    if scenario == "Stationary Exponential":
        return "Exponential"

    if scenario == "Stationary Uniform":
        return "Uniform"

    if frame < SWITCH_1:
        return "Exponential"

    if frame < SWITCH_2:
        return "Uniform"

    return "Exponential"


def phase_name(scenario, frame):

    if scenario != "Dynamic E-U-E":
        return "Stationary"

    if frame < SWITCH_1:
        return "Phase 1"

    if frame < SWITCH_2:
        return "Phase 2"

    return "Phase 3"


# ============================================================
# 5. PRECOMPUTE ESTIMATION AND LIKELIHOOD MATRICES
# ============================================================

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
            raise ValueError(
                f"Invalid covariance matrix for {name}"
            )

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
# 6. EWMA DETECTOR
# ============================================================

class EWMADetector:

    def __init__(self, initial_profile):

        self.alpha = EWMA_ALPHA

        self.score = 0.0
        self.initialized = False

        self.selected = initial_profile
        self.candidate = None
        self.count = 0

    def update(self, difference):

        if not self.initialized:

            self.score = difference
            self.initialized = True

        else:

            self.score = (
                self.alpha * self.score
                + (1 - self.alpha) * difference
            )

        proposed = (
            "Uniform"
            if self.score > 0
            else "Exponential"
        )

        switched = False
        old_profile = self.selected

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

                # Reset accumulated evidence after switching.
                self.score = 0.0
                self.initialized = False

        return self.selected, switched, old_profile


# ============================================================
# 7. SLIDING-WINDOW DETECTOR
# ============================================================

class WindowDetector:

    def __init__(self, initial_profile):

        self.window = deque(maxlen=WINDOW_LENGTH)

        self.selected = initial_profile
        self.candidate = None
        self.count = 0

    def update(self, difference):

        self.window.append(difference)

        switched = False
        old_profile = self.selected

        if len(self.window) < WINDOW_LENGTH:

            return self.selected, switched, old_profile

        mean_difference = float(
            np.mean(self.window)
        )

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

                # Remove evidence belonging to the old regime.
                self.window.clear()

        return self.selected, switched, old_profile


# ============================================================
# 8. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(seed, snr_db, scenario):

    rng = np.random.default_rng(seed)

    noise_variance = 10 ** (-snr_db / 10)

    models = prepare_models(noise_variance)

    # Stationary tests start with their known correct profile.
    # Dynamic tests start with Exponential.
    if scenario == "Stationary Uniform":

        initial_profile = "Uniform"

    else:

        initial_profile = "Exponential"

    detectors = {
        "EWMA": EWMADetector(initial_profile),
        "Sliding Window": WindowDetector(initial_profile)
    }

    methods = [
        "Perfect CSI",
        "LS",
        "Fixed Exponential LMMSE",
        "Fixed Uniform LMMSE",
        "EWMA Adaptive LMMSE",
        "Window Adaptive LMMSE"
    ]

    if scenario == "Dynamic E-U-E":

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

    # ========================================================
    # FRAME LOOP
    # ========================================================

    for frame in range(NUM_FRAMES):

        true_profile_name = actual_profile(
            scenario, frame
        )

        phase = phase_name(scenario, frame)

        # Generate QPSK information bits.
        bits = rng.integers(
            0, 2,
            len(DATA) * 2,
            dtype=np.int8
        )

        symbols = qpsk_modulate(bits)

        # Build OFDM frequency-domain symbol.
        frequency_symbol = np.zeros(
            NUM_SUBCARRIERS,
            dtype=complex
        )

        frequency_symbol[PILOTS] = 1.0 + 0j
        frequency_symbol[DATA] = symbols

        # IFFT and cyclic prefix.
        time_symbol = (
            np.fft.ifft(frequency_symbol)
            * np.sqrt(NUM_SUBCARRIERS)
        )

        transmitted = np.concatenate([
            time_symbol[-CP_LENGTH:],
            time_symbol
        ])

        # Generate channel.
        powers = PROFILES[true_profile_name]

        taps = (
            rng.normal(size=NUM_TAPS)
            + 1j * rng.normal(size=NUM_TAPS)
        ) / np.sqrt(2)

        taps *= np.sqrt(powers)

        true_channel = np.fft.fft(
            taps,
            NUM_SUBCARRIERS
        )

        # Apply multipath channel and AWGN.
        received = np.convolve(
            transmitted,
            taps
        )

        noise = np.sqrt(noise_variance / 2) * (
            rng.normal(size=len(received))
            + 1j * rng.normal(size=len(received))
        )

        received = received + noise

        # Remove cyclic prefix.
        received_no_cp = received[
            CP_LENGTH:
            CP_LENGTH + NUM_SUBCARRIERS
        ]

        # FFT.
        received_frequency = (
            np.fft.fft(received_no_cp)
            / np.sqrt(NUM_SUBCARRIERS)
        )

        pilot_values = received_frequency[PILOTS]
        received_data = received_frequency[DATA]

        # ====================================================
        # CHANNEL ESTIMATION
        # ====================================================

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

        # Adaptive estimates use the model selected
        # before processing the current frame.
        for detector_name, detector in detectors.items():

            selected_profile = detector.selected

            adaptive_estimate = estimate_channel(
                pilot_values,
                models[selected_profile]
            )

            if detector_name == "EWMA":

                method = "EWMA Adaptive LMMSE"

            else:

                method = "Window Adaptive LMMSE"

            estimates[method] = adaptive_estimate

            if selected_profile == true_profile_name:

                selection_correct[detector_name][phase] += 1

            histories[detector_name].append({
                "Frame": frame + 1,
                "Actual Profile": true_profile_name,
                "Selected Profile": selected_profile,
                "Phase": phase
            })

        # ====================================================
        # BER AND NMSE
        # ====================================================

        true_power = np.mean(
            np.abs(true_channel) ** 2
        )

        for method, estimate in estimates.items():

            channel_data = estimate[DATA]

            # Numerically stable zero-forcing equalization.
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

        # ====================================================
        # UPDATE DETECTORS AFTER CURRENT-FRAME ESTIMATION
        # ====================================================

        nll_exp = calculate_nll(
            pilot_values,
            models["Exponential"]
        )

        nll_uni = calculate_nll(
            pilot_values,
            models["Uniform"]
        )

        # Positive difference favours Uniform.
        difference = nll_exp - nll_uni

        for detector_name, detector in detectors.items():

            new_selection, switched, old_profile = (
                detector.update(difference)
            )

            if switched:

                # Current frame is frame+1.
                # The new decision is used from frame+2.
                effective_frame = frame + 2

                switch_events[detector_name].append({
                    "Decision Frame": frame + 1,
                    "Effective Frame": effective_frame,
                    "Old Profile": old_profile,
                    "New Profile": new_selection
                })

    # ========================================================
    # BUILD METRIC RESULTS
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

            result_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Scenario": scenario,
                "Phase": phase,
                "Method": (
                    f"{detector_name} Selection Accuracy"
                ),
                "BER": np.nan,
                "NMSE": np.nan,
                "Selection Accuracy": (
                    selection_correct[detector_name][phase]
                    / frame_counts[phase]
                )
            })

    # ========================================================
    # FALSE SWITCHES AND DETECTION DELAYS
    # ========================================================

    event_rows = []

    for detector_name, events in switch_events.items():

        # Count switches that lead to an incorrect model
        # on the frame where that model first takes effect.
        false_events = []

        for event in events:

            effective_frame = event["Effective Frame"]

            true_profile_at_use = actual_profile(
                scenario,
                effective_frame - 1
            )

            if event["New Profile"] != true_profile_at_use:

                false_events.append(event)

        false_count = len(false_events)

        if scenario == "Dynamic E-U-E":

            expected_switches = [
                (
                    SWITCH_1 + 1,
                    "Exponential",
                    "Uniform"
                ),
                (
                    SWITCH_2 + 1,
                    "Uniform",
                    "Exponential"
                )
            ]

            detector_history = histories[detector_name]

            for switch_frame, old_profile, new_profile in expected_switches:

                # Find the first frame after the actual
                # channel change where the correct new
                # profile is actually used.
                correct_frames = [
                    record["Frame"]
                    for record in detector_history
                    if (
                        record["Frame"] >= switch_frame
                        and record["Selected Profile"] == new_profile
                    )
                ]

                if correct_frames:

                    detected_frame = min(correct_frames)

                    detection_delay = (
                        detected_frame - switch_frame
                    )

                else:

                    detected_frame = -1
                    detection_delay = -1

                event_rows.append({
                    "Seed": seed,
                    "SNR (dB)": snr_db,
                    "Scenario": scenario,
                    "Detector": detector_name,
                    "Event": (
                        f"{old_profile} to {new_profile}"
                    ),
                    "Detection Frame": detected_frame,
                    "Detection Delay": detection_delay,
                    "False Switches": np.nan
                })

        event_rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Scenario": scenario,
            "Detector": detector_name,
            "Event": (
                "Total False Switches"
                if scenario == "Dynamic E-U-E"
                else "Stationary False Switches"
            ),
            "Detection Frame": np.nan,
            "Detection Delay": np.nan,
            "False Switches": false_count
        })

    # ========================================================
    # SELECTION HISTORY
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

    return result_rows, event_rows, history_rows


# ============================================================
# 9. MAIN EXPERIMENT
# ============================================================

def main():

    all_results = []
    all_events = []
    all_history = []

    total_runs = (
        len(SEEDS)
        * len(SNR_VALUES)
        * len(SCENARIOS)
    )

    completed = 0

    print("\nCHANNEL DETECTOR ROBUSTNESS VALIDATION")
    print("--------------------------------------")
    print("Number of seeds:", len(SEEDS))
    print("SNR values:", SNR_VALUES)
    print("Scenarios:", SCENARIOS)
    print("EWMA alpha:", EWMA_ALPHA)
    print("Sliding window:", WINDOW_LENGTH)
    print("Confirmation frames:", CONFIRMATION_FRAMES)
    print("Total experiments:", total_runs)

    for scenario in SCENARIOS:

        for snr_db in SNR_VALUES:

            for seed in SEEDS:

                results, events, history = run_experiment(
                    seed,
                    snr_db,
                    scenario
                )

                all_results.extend(results)
                all_events.extend(events)
                all_history.extend(history)

                completed += 1

                print(
                    f"Progress: {completed}/{total_runs} | "
                    f"{scenario} | "
                    f"SNR: {snr_db} dB | "
                    f"Seed: {seed}"
                )

    raw = pd.DataFrame(all_results)
    event_df = pd.DataFrame(all_events)
    history_df = pd.DataFrame(all_history)

    # ========================================================
    # SAVE RAW DATA
    # ========================================================

    raw.to_csv(
        RESULTS_DIR / "detector_robustness_raw.csv",
        index=False
    )

    event_df.to_csv(
        RESULTS_DIR / "detector_robustness_events.csv",
        index=False
    )

    history_df.to_csv(
        RESULTS_DIR / "detector_robustness_history.csv",
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

    summary["BER_CI_Lower"] = np.maximum(
        0,
        summary["Mean_BER"]
        - T_CRITICAL * summary["Std_BER"]
        / np.sqrt(len(SEEDS))
    )

    summary["BER_CI_Upper"] = np.minimum(
        1,
        summary["Mean_BER"]
        + T_CRITICAL * summary["Std_BER"]
        / np.sqrt(len(SEEDS))
    )

    summary.to_csv(
        RESULTS_DIR / "detector_robustness_summary.csv",
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
        RESULTS_DIR / "detector_accuracy_summary.csv",
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
            Mean_False_Switches=("False Switches", "mean"),
            Std_False_Switches=("False Switches", "std")
        )
        .reset_index()
    )

    false_summary.to_csv(
        RESULTS_DIR / "detector_false_switch_summary.csv",
        index=False
    )

    # ========================================================
    # DETECTION DELAY SUMMARY
    # ========================================================

    delay_df = event_df[
        (event_df["Scenario"] == "Dynamic E-U-E")
        & event_df["Detection Delay"].notna()
    ]

    delay_summary = (
        delay_df.groupby([
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

    delay_summary.to_csv(
        RESULTS_DIR / "detector_delay_summary.csv",
        index=False
    )

    # ========================================================
    # BER GRAPHS
    # ========================================================

    for phase in ["Phase 1", "Phase 2", "Phase 3"]:

        plt.figure(figsize=(10, 6))

        subset = summary[
            (summary["Scenario"] == "Dynamic E-U-E")
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
            f"Channel Detector BER Comparison - {phase}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = (
            f"detector_ber_"
            f"{phase.lower().replace(' ', '_')}.png"
        )

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300
        )

        plt.close()

    # ========================================================
    # FALSE SWITCH GRAPH
    # ========================================================

    for scenario in SCENARIOS:

        plt.figure(figsize=(9, 5))

        subset = false_summary[
            false_summary["Scenario"] == scenario
        ]

        for detector in ["EWMA", "Sliding Window"]:

            data = subset[
                subset["Detector"] == detector
            ].sort_values("SNR (dB)")

            plt.plot(
                data["SNR (dB)"],
                data["Mean_False_Switches"],
                marker="o",
                label=detector
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean False Switches per Run")
        plt.title(
            f"False Switching - {scenario}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = (
            "detector_false_switch_"
            + scenario.lower().replace(" ", "_")
            + ".png"
        )

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300
        )

        plt.close()

    # ========================================================
    # FINAL TERMINAL REPORT
    # ========================================================

    print("\n======================================")
    print("ROBUSTNESS VALIDATION COMPLETED")
    print("======================================")

    print("\nFALSE SWITCH SUMMARY:")
    print(false_summary.to_string(index=False))

    print("\nDETECTION DELAY SUMMARY:")
    print(delay_summary.to_string(index=False))

    print("\nDYNAMIC ADAPTIVE BER SUMMARY:")

    adaptive = summary[
        (summary["Scenario"] == "Dynamic E-U-E")
        & summary["Method"].isin([
            "EWMA Adaptive LMMSE",
            "Window Adaptive LMMSE"
        ])
    ]

    print(
        adaptive[
            [
                "SNR (dB)",
                "Phase",
                "Method",
                "Mean_BER",
                "Mean_NMSE"
            ]
        ].to_string(index=False)
    )

    print("\nSELECTION ACCURACY SUMMARY:")

    print(
        accuracy_summary.to_string(index=False)
    )

    print("\nSaved experiment files in results/")

    print("\nImportant:")
    print("- Stationary tests use correct initial profiles.")
    print("- Dynamic tests start with Exponential.")
    print("- Detection delay uses the effective selection frame.")
    print("- False switches are evaluated against the actual profile.")


if __name__ == "__main__":
    main()
