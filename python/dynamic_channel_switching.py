
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# 1. CONFIGURATION
# =========================================================

SEEDS = [101, 202, 303, 404, 505]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
NUM_TAPS = 8
CP_LENGTH = 16

PILOT_SPACING = 4
PILOTS = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA = np.setdiff1d(
    np.arange(NUM_SUBCARRIERS), PILOTS
)

SNR_VALUES = [0, 5, 10, 15, 20]

FORGETTING_FACTOR = 0.95
CONFIRMATION_FRAMES = 5

SWITCH_1 = 300
SWITCH_2 = 600

PHASES = {
    "Phase 1": (0, SWITCH_1, "Exponential"),
    "Phase 2": (SWITCH_1, SWITCH_2, "Uniform"),
    "Phase 3": (SWITCH_2, NUM_FRAMES, "Exponential")
}

PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for name in PROFILES:
    PROFILES[name] /= np.sum(PROFILES[name])

ESTIMATORS = [
    "Perfect CSI",
    "LS",
    "Fixed Exponential LMMSE",
    "Fixed Uniform LMMSE",
    "Online Adaptive LMMSE"
]

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# =========================================================
# 2. QPSK MODULATION
# =========================================================

def qpsk_modulate(bits):

    pairs = bits.reshape(-1, 2)

    return (
        (1 - 2 * pairs[:, 0])
        + 1j * (1 - 2 * pairs[:, 1])
    ) / np.sqrt(2)


def qpsk_demodulate(symbols):

    bits = np.empty(
        (len(symbols), 2),
        dtype=np.int8
    )

    bits[:, 0] = (symbols.real < 0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0).astype(np.int8)

    return bits.reshape(-1)


# =========================================================
# 3. CHANNEL COVARIANCE
# =========================================================

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


# =========================================================
# 4. CHANNEL GENERATION
# =========================================================

def generate_channel(rng, profile_name):

    powers = PROFILES[profile_name]

    taps = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    taps *= np.sqrt(powers)

    frequency_response = np.fft.fft(
        taps, NUM_SUBCARRIERS
    )

    return taps, frequency_response


# =========================================================
# 5. LMMSE ESTIMATION
# =========================================================

def lmmse_estimate(
    pilot_values,
    noise_variance,
    covariance
):

    pilot_covariance = covariance[
        np.ix_(PILOTS, PILOTS)
    ]

    cross_covariance = covariance[:, PILOTS]

    matrix = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOTS))
    )

    weights = np.linalg.solve(
        matrix,
        pilot_values
    )

    return cross_covariance @ weights


# =========================================================
# 6. NEGATIVE LOG-LIKELIHOOD
# =========================================================

def profile_negative_log_likelihood(
    pilot_values,
    noise_variance,
    covariance
):

    pilot_covariance = covariance[
        np.ix_(PILOTS, PILOTS)
    ]

    observation_covariance = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOTS))
    )

    sign, logdet = np.linalg.slogdet(
        observation_covariance
    )

    if sign <= 0:
        raise ValueError(
            "Invalid covariance matrix"
        )

    solution = np.linalg.solve(
        observation_covariance,
        pilot_values
    )

    quadratic_term = np.real(
        np.vdot(pilot_values, solution)
    )

    return float(logdet + quadratic_term)


# =========================================================
# 7. LS INTERPOLATION
# =========================================================

def ls_estimate(pilot_values):

    return (
        np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOTS,
            pilot_values.real
        )
        + 1j * np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOTS,
            pilot_values.imag
        )
    )


# =========================================================
# 8. ACTUAL CHANNEL PROFILE AT EACH FRAME
# =========================================================

def get_actual_profile(frame):

    if frame < SWITCH_1:
        return "Exponential"

    elif frame < SWITCH_2:
        return "Uniform"

    return "Exponential"


def get_phase(frame):

    if frame < SWITCH_1:
        return "Phase 1"

    elif frame < SWITCH_2:
        return "Phase 2"

    return "Phase 3"


# =========================================================
# 9. SWITCH DETECTION DELAY
# =========================================================

def calculate_detection_delay(
    selected_history,
    switch_frame,
    expected_profile
):

    last_possible_start = (
        len(selected_history)
        - CONFIRMATION_FRAMES
    )

    for start in range(
        switch_frame,
        last_possible_start + 1
    ):

        window = selected_history[
            start:start + CONFIRMATION_FRAMES
        ]

        if all(
            profile == expected_profile
            for profile in window
        ):

            return start - switch_frame

    # -1 means the switch was not confirmed
    return -1


# =========================================================
# 10. ONE COMPLETE SIMULATION RUN
# =========================================================

def run_experiment(seed, snr_db):

    rng = np.random.default_rng(seed)

    noise_variance = 10 ** (-snr_db / 10)

    errors = {
        phase: {
            estimator: 0
            for estimator in ESTIMATORS
        }
        for phase in PHASES
    }

    bit_counts = {
        phase: 0
        for phase in PHASES
    }

    nmse_sums = {
        phase: {
            estimator: 0.0
            for estimator in ESTIMATORS
            if estimator != "Perfect CSI"
        }
        for phase in PHASES
    }

    phase_frame_counts = {
        phase: 0
        for phase in PHASES
    }

    accumulated_scores = {
        "Exponential": 0.0,
        "Uniform": 0.0
    }

    # Initial profile assumption
    selected_profile = "Exponential"

    selected_used_history = []
    selected_next_history = []
    actual_history = []

    frame_rows = []

    # -----------------------------------------------------
    # Process OFDM frames
    # -----------------------------------------------------

    for frame in range(NUM_FRAMES):

        actual_profile = get_actual_profile(frame)
        phase = get_phase(frame)

        # Generate QPSK data
        bits = rng.integers(
            0,
            2,
            len(DATA) * 2,
            dtype=np.int8
        )

        symbols = qpsk_modulate(bits)

        frequency_symbol = np.zeros(
            NUM_SUBCARRIERS,
            dtype=complex
        )

        frequency_symbol[PILOTS] = 1 + 0j
        frequency_symbol[DATA] = symbols

        # OFDM modulation
        time_symbol = (
            np.fft.ifft(frequency_symbol)
            * np.sqrt(NUM_SUBCARRIERS)
        )

        with_cp = np.concatenate([
            time_symbol[-CP_LENGTH:],
            time_symbol
        ])

        # Generate channel for current frame
        channel_taps, true_channel = generate_channel(
            rng,
            actual_profile
        )

        received = np.convolve(
            with_cp,
            channel_taps
        )

        # AWGN
        noise = np.sqrt(noise_variance / 2) * (
            rng.normal(size=len(received))
            + 1j * rng.normal(size=len(received))
        )

        received += noise

        # Remove cyclic prefix
        received_no_cp = received[
            CP_LENGTH:
            CP_LENGTH + NUM_SUBCARRIERS
        ]

        # OFDM demodulation
        received_frequency = (
            np.fft.fft(received_no_cp)
            / np.sqrt(NUM_SUBCARRIERS)
        )

        pilot_values = received_frequency[PILOTS]
        received_data = received_frequency[DATA]

        # -------------------------------------------------
        # A. USE PREVIOUSLY SELECTED MODEL
        # -------------------------------------------------

        selected_used = selected_profile

        online_estimate = lmmse_estimate(
            pilot_values,
            noise_variance,
            COVARIANCES[selected_used]
        )

        # Fixed estimators
        ls = ls_estimate(pilot_values)

        exponential = lmmse_estimate(
            pilot_values,
            noise_variance,
            COVARIANCES["Exponential"]
        )

        uniform = lmmse_estimate(
            pilot_values,
            noise_variance,
            COVARIANCES["Uniform"]
        )

        # Equalization
        equalized = {
            "Perfect CSI":
                received_data / true_channel[DATA],

            "LS":
                received_data / ls[DATA],

            "Fixed Exponential LMMSE":
                received_data / exponential[DATA],

            "Fixed Uniform LMMSE":
                received_data / uniform[DATA],

            "Online Adaptive LMMSE":
                received_data / online_estimate[DATA]
        }

        # -------------------------------------------------
        # B. BER CALCULATION
        # -------------------------------------------------

        for estimator in ESTIMATORS:

            detected = qpsk_demodulate(
                equalized[estimator]
            )

            errors[phase][estimator] += (
                np.count_nonzero(detected != bits)
            )

        bit_counts[phase] += len(bits)
        phase_frame_counts[phase] += 1

        # -------------------------------------------------
        # C. NMSE CALCULATION
        # -------------------------------------------------

        true_power = np.mean(
            np.abs(true_channel) ** 2
        )

        estimates = {
            "LS": ls,
            "Fixed Exponential LMMSE": exponential,
            "Fixed Uniform LMMSE": uniform,
            "Online Adaptive LMMSE": online_estimate
        }

        for estimator, estimate in estimates.items():

            frame_nmse = (
                np.mean(
                    np.abs(estimate - true_channel) ** 2
                ) / true_power
            )

            nmse_sums[phase][estimator] += frame_nmse

        # -------------------------------------------------
        # D. RECORD PROFILE USED FOR THIS FRAME
        # -------------------------------------------------

        selected_used_history.append(
            selected_used
        )

        actual_history.append(
            actual_profile
        )

        # -------------------------------------------------
        # E. UPDATE MODEL LIKELIHOODS
        # AFTER CURRENT FRAME ESTIMATION
        # -------------------------------------------------

        for profile_name in PROFILES:

            current_score = (
                profile_negative_log_likelihood(
                    pilot_values,
                    noise_variance,
                    COVARIANCES[profile_name]
                )
            )

            accumulated_scores[profile_name] = (
                FORGETTING_FACTOR
                * accumulated_scores[profile_name]
                + (1 - FORGETTING_FACTOR)
                * current_score
            )

        # The selected profile is used from the NEXT frame
        selected_profile = min(
            accumulated_scores,
            key=accumulated_scores.get
        )

        selected_next_history.append(
            selected_profile
        )

        frame_rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Frame": frame + 1,
            "Phase": phase,
            "Actual Profile": actual_profile,
            "Profile Used": selected_used,
            "Profile Selected Next": selected_profile
        })

    # -----------------------------------------------------
    # F. PHASE-LEVEL RESULTS
    # -----------------------------------------------------

    run_rows = []

    for phase in PHASES:

        for estimator in ESTIMATORS:

            ber = (
                errors[phase][estimator]
                / bit_counts[phase]
            )

            if estimator == "Perfect CSI":
                nmse = np.nan
            else:
                nmse = (
                    nmse_sums[phase][estimator]
                    / phase_frame_counts[phase]
                )

            run_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Phase": phase,
                "Actual Profile": PHASES[phase][2],
                "Estimator": estimator,
                "BER": ber,
                "NMSE": nmse
            })

    # -----------------------------------------------------
    # G. PROFILE SELECTION ACCURACY
    # -----------------------------------------------------

    for phase, (start, end, expected) in PHASES.items():

        correct = sum(
            selected_used_history[i] == expected
            for i in range(start, end)
        )

        accuracy = correct / (end - start)

        run_rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Phase": phase,
            "Actual Profile": expected,
            "Estimator": "Profile Selection Accuracy",
            "BER": np.nan,
            "NMSE": np.nan,
            "Selection Accuracy": accuracy
        })

    # -----------------------------------------------------
    # H. DETECTION DELAY
    # -----------------------------------------------------

    delay_1 = calculate_detection_delay(
        selected_used_history,
        SWITCH_1,
        "Uniform"
    )

    delay_2 = calculate_detection_delay(
        selected_used_history,
        SWITCH_2,
        "Exponential"
    )

    detection_rows = [
        {
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Switch": "Exponential to Uniform",
            "Switch Frame": SWITCH_1 + 1,
            "Detection Delay (frames)": delay_1
        },
        {
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Switch": "Uniform to Exponential",
            "Switch Frame": SWITCH_2 + 1,
            "Detection Delay (frames)": delay_2
        }
    ]

    return (
        run_rows,
        frame_rows,
        detection_rows
    )


# =========================================================
# 11. MAIN EXPERIMENT
# =========================================================

def main():

    all_results = []
    all_frames = []
    all_detection = []

    total_runs = len(SEEDS) * len(SNR_VALUES)
    completed = 0

    print("\nDYNAMIC CHANNEL PROFILE SWITCHING")
    print("---------------------------------")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Frames per run: {NUM_FRAMES}")
    print(f"Forgetting factor: {FORGETTING_FACTOR}")
    print(f"SNR values: {SNR_VALUES}")

    for snr_db in SNR_VALUES:

        for seed in SEEDS:

            (
                run_rows,
                frame_rows,
                detection_rows
            ) = run_experiment(seed, snr_db)

            all_results.extend(run_rows)
            all_frames.extend(frame_rows)
            all_detection.extend(detection_rows)

            completed += 1

            print(
                f"Progress: {completed}/{total_runs} | "
                f"SNR: {snr_db} dB | Seed: {seed}"
            )

    raw_df = pd.DataFrame(all_results)
    frame_df = pd.DataFrame(all_frames)
    detection_df = pd.DataFrame(all_detection)

    # -----------------------------------------------------
    # SAVE RAW DATA
    # -----------------------------------------------------

    raw_df.to_csv(
        RESULTS / "dynamic_channel_raw_results.csv",
        index=False
    )

    frame_df.to_csv(
        RESULTS / "dynamic_channel_frame_history.csv",
        index=False
    )

    detection_df.to_csv(
        RESULTS / "dynamic_channel_detection_delays.csv",
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    metric_df = raw_df[
        raw_df["Estimator"] != "Profile Selection Accuracy"
    ]

    summary = (
        metric_df.groupby(
            ["SNR (dB)", "Phase", "Actual Profile", "Estimator"]
        )
        .agg(
            Mean_BER=("BER", "mean"),
            Std_BER=("BER", "std"),
            Mean_NMSE=("NMSE", "mean")
        )
        .reset_index()
    )

    # Preliminary 95% t confidence intervals, five seeds
    t_value = 2.776

    margin = (
        t_value
        * summary["Std_BER"]
        / np.sqrt(len(SEEDS))
    )

    summary["BER_CI_Lower"] = np.maximum(
        0,
        summary["Mean_BER"] - margin
    )

    summary["BER_CI_Upper"] = np.minimum(
        1,
        summary["Mean_BER"] + margin
    )

    summary.to_csv(
        RESULTS / "dynamic_channel_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # PROFILE ACCURACY SUMMARY
    # -----------------------------------------------------

    accuracy_df = raw_df[
        raw_df["Estimator"] == "Profile Selection Accuracy"
    ]

    accuracy_summary = (
        accuracy_df.groupby(
            ["SNR (dB)", "Phase", "Actual Profile"]
        )
        .agg(
            Mean_Accuracy=("Selection Accuracy", "mean"),
            Std_Accuracy=("Selection Accuracy", "std")
        )
        .reset_index()
    )

    accuracy_summary.to_csv(
        RESULTS / "dynamic_channel_accuracy_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # DETECTION DELAY SUMMARY
    # -1 means not detected within the run.
    # -----------------------------------------------------

    detection_summary = (
        detection_df.groupby(
            ["SNR (dB)", "Switch"]
        )
        .agg(
            Mean_Delay=("Detection Delay (frames)", "mean"),
            Detection_Rate=(
                "Detection Delay (frames)",
                lambda x: np.mean(x >= 0)
            )
        )
        .reset_index()
    )

    detection_summary.to_csv(
        RESULTS / "dynamic_channel_detection_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # BER PLOTS BY PHASE
    # -----------------------------------------------------

    for phase in PHASES:

        plt.figure(figsize=(10, 6))

        phase_data = summary[
            summary["Phase"] == phase
        ]

        for estimator in ESTIMATORS:

            subset = phase_data[
                phase_data["Estimator"] == estimator
            ].sort_values("SNR (dB)")

            plt.errorbar(
                subset["SNR (dB)"],
                subset["Mean_BER"],
                yerr=[
                    subset["Mean_BER"]
                    - subset["BER_CI_Lower"],
                    subset["BER_CI_Upper"]
                    - subset["Mean_BER"]
                ],
                marker="o",
                capsize=3,
                label=estimator
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean BER")
        plt.title(
            f"Dynamic Channel BER - {phase}"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS / f"dynamic_channel_ber_{phase.lower().replace(' ', '_')}.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # PROFILE SELECTION ACCURACY GRAPH
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    for phase in PHASES:

        subset = accuracy_summary[
            accuracy_summary["Phase"] == phase
        ].sort_values("SNR (dB)")

        plt.plot(
            subset["SNR (dB)"],
            subset["Mean_Accuracy"] * 100,
            marker="o",
            label=phase
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Profile Selection Accuracy (%)")
    plt.title("Dynamic Channel Profile Selection Accuracy")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / "dynamic_channel_selection_accuracy.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # EXAMPLE PROFILE SWITCHING GRAPH
    # Seed 101, SNR 10 dB
    # -----------------------------------------------------

    example = frame_df[
        (frame_df["Seed"] == 101)
        & (frame_df["SNR (dB)"] == 10)
    ]

    frame_numbers = example["Frame"].to_numpy()

    actual_numeric = example[
        "Actual Profile"
    ].map({
        "Exponential": 0,
        "Uniform": 1
    }).to_numpy()

    selected_numeric = example[
        "Profile Used"
    ].map({
        "Exponential": 0,
        "Uniform": 1
    }).to_numpy()

    plt.figure(figsize=(12, 5))

    plt.step(
        frame_numbers,
        actual_numeric,
        where="post",
        label="Actual Channel Profile"
    )

    plt.step(
        frame_numbers,
        selected_numeric,
        where="post",
        label="Selected Profile Used"
    )

    plt.axvline(
        SWITCH_1,
        linestyle="--",
        label="First Channel Switch"
    )

    plt.axvline(
        SWITCH_2,
        linestyle="--",
        label="Second Channel Switch"
    )

    plt.yticks(
        [0, 1],
        ["Exponential", "Uniform"]
    )

    plt.xlabel("OFDM Frame")
    plt.ylabel("Channel Profile")
    plt.title(
        "Online Profile Tracking (Seed 101, SNR 10 dB)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / "dynamic_channel_profile_tracking.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    print("\n---------------------------------")
    print("DYNAMIC CHANNEL EXPERIMENT DONE")
    print("---------------------------------")

    print("\nBER AND NMSE SUMMARY:")

    print(
        summary[
            [
                "SNR (dB)",
                "Phase",
                "Estimator",
                "Mean_BER",
                "Mean_NMSE"
            ]
        ].to_string(index=False)
    )

    print("\nPROFILE SELECTION ACCURACY:")

    print(
        accuracy_summary.to_string(index=False)
    )

    print("\nCHANNEL SWITCH DETECTION:")

    print(
        detection_summary.to_string(index=False)
    )

    print("\nSaved results in results/")

    print("dynamic_channel_raw_results.csv")
    print("dynamic_channel_frame_history.csv")
    print("dynamic_channel_detection_delays.csv")
    print("dynamic_channel_summary.csv")
    print("dynamic_channel_accuracy_summary.csv")
    print("dynamic_channel_detection_summary.csv")
    print("dynamic_channel_profile_tracking.png")


if __name__ == "__main__":
    main()
