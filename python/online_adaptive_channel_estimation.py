
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

SEEDS = [101, 202, 303, 404, 505]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
NUM_TAPS = 8
CP_LENGTH = 16

PILOT_SPACING = 4
PILOTS = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA = np.setdiff1d(np.arange(NUM_SUBCARRIERS), PILOTS)

SNR_VALUES = [0, 5, 10, 15, 20]

# Exponentially weighted online likelihood
FORGETTING_FACTOR = 0.98

PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for name in PROFILES:
    PROFILES[name] = PROFILES[name] / np.sum(PROFILES[name])

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

ESTIMATORS = [
    "Perfect CSI",
    "LS",
    "Fixed Exponential LMMSE",
    "Fixed Uniform LMMSE",
    "Online Adaptive LMMSE"
]


# =====================================================
# MODULATION
# =====================================================

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


# =====================================================
# CHANNEL COVARIANCE
# =====================================================

def calculate_covariance(profile):

    indices = np.arange(NUM_SUBCARRIERS)

    difference = indices[:, None] - indices[None, :]

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


# =====================================================
# CHANNEL GENERATION
# =====================================================

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


# =====================================================
# LMMSE ESTIMATION
# =====================================================

def lmmse_estimate(pilot_values, noise_variance, covariance):

    pilot_covariance = covariance[
        np.ix_(PILOTS, PILOTS)
    ]

    cross_covariance = covariance[:, PILOTS]

    matrix = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOTS))
    )

    weights = np.linalg.solve(matrix, pilot_values)

    return cross_covariance @ weights


# =====================================================
# CHANNEL MODEL LIKELIHOOD
# =====================================================

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
        raise ValueError("Invalid covariance matrix")

    solution = np.linalg.solve(
        observation_covariance,
        pilot_values
    )

    quadratic_term = np.real(
        np.vdot(pilot_values, solution)
    )

    return float(logdet + quadratic_term)


# =====================================================
# LS INTERPOLATION
# =====================================================

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


# =====================================================
# ONE SIMULATION RUN
# =====================================================

def run_experiment(seed, snr_db, actual_profile):

    rng = np.random.default_rng(seed)

    noise_variance = 10 ** (-snr_db / 10)

    errors = {name: 0 for name in ESTIMATORS}

    nmse_sum = {
        name: 0.0
        for name in ESTIMATORS
        if name != "Perfect CSI"
    }

    # Online accumulated model scores
    accumulated_scores = {
        "Exponential": 0.0,
        "Uniform": 0.0
    }

    # No historical observations at the start.
    # Use a deterministic default model for frame 0.
    selected_profile = "Exponential"

    selection_correct = 0
    selection_history = []

    total_bits = 0

    for frame in range(NUM_FRAMES):

        # Generate data
        bits = rng.integers(
            0, 2, len(DATA) * 2, dtype=np.int8
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

        # Actual channel
        channel_taps, true_channel = generate_channel(
            rng, actual_profile
        )

        received = np.convolve(with_cp, channel_taps)

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
        # Estimate using previous observations
        # -------------------------------------------------

        online_estimate = lmmse_estimate(
            pilot_values,
            noise_variance,
            COVARIANCES[selected_profile]
        )
        # Record the profile actually used for this frame
        if selected_profile == actual_profile:
            selection_correct += 1

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

        # Perfect CSI
        perfect_equalized = (
            received_data / true_channel[DATA]
        )

        # Equalization
        equalized = {
            "Perfect CSI": perfect_equalized,

            "LS": received_data / ls[DATA],

            "Fixed Exponential LMMSE":
                received_data / exponential[DATA],

            "Fixed Uniform LMMSE":
                received_data / uniform[DATA],

            "Online Adaptive LMMSE":
                received_data / online_estimate[DATA]
        }

        # Detection
        for estimator in ESTIMATORS:

            detected = qpsk_demodulate(
                equalized[estimator]
            )

            errors[estimator] += np.count_nonzero(
                detected != bits
            )

        total_bits += len(bits)

        # NMSE
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

            nmse_sum[estimator] += (
                np.mean(
                    np.abs(estimate - true_channel) ** 2
                ) / true_power
            )

        # -------------------------------------------------
        # Update online scores AFTER current-frame
        # estimation and detection.
        # This prevents current-frame model-selection
        # information from leaking into its own estimate.
        # -------------------------------------------------

        for profile_name in PROFILES:

            current_score = profile_negative_log_likelihood(
                pilot_values,
                noise_variance,
                COVARIANCES[profile_name]
            )

            accumulated_scores[profile_name] = (
                FORGETTING_FACTOR
                * accumulated_scores[profile_name]
                + (1 - FORGETTING_FACTOR) * current_score
            )

        # Selection will be used from the NEXT frame
        selected_profile = min(
            accumulated_scores,
            key=accumulated_scores.get
        )

        selection_history.append(selected_profile)

    # -----------------------------------------------------
    # Return results
    # -----------------------------------------------------

    rows = []

    for estimator in ESTIMATORS:

        ber = errors[estimator] / total_bits

        if estimator == "Perfect CSI":
            nmse = np.nan
        else:
            nmse = nmse_sum[estimator] / NUM_FRAMES

        rows.append({
            "Seed": seed,
            "Actual Channel": actual_profile,
            "SNR (dB)": snr_db,
            "Estimator": estimator,
            "BER": ber,
            "NMSE": nmse,
            "Online Selection Accuracy": (
                selection_correct / NUM_FRAMES
            )
        })

    return rows, selection_history


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def main():

    all_rows = []
    all_selection_rows = []

    total_runs = (
        len(SEEDS) * len(SNR_VALUES) * len(PROFILES)
    )

    completed = 0

    print("\nONLINE ADAPTIVE OFDM CHANNEL ESTIMATION")
    print("---------------------------------------")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Frames per run: {NUM_FRAMES}")
    print(f"Forgetting factor: {FORGETTING_FACTOR}")
    print(f"SNR values: {SNR_VALUES}")

    for actual_profile in PROFILES:

        for snr_db in SNR_VALUES:

            for seed in SEEDS:

                rows, history = run_experiment(
                    seed,
                    snr_db,
                    actual_profile
                )

                all_rows.extend(rows)

                for frame, selected in enumerate(history):

                    all_selection_rows.append({
                        "Seed": seed,
                        "Actual Channel": actual_profile,
                        "SNR (dB)": snr_db,
                        "Frame": frame,
                        "Selected Channel": selected
                    })

                completed += 1

                print(
                    f"Progress: {completed}/{total_runs} | "
                    f"Channel: {actual_profile} | "
                    f"SNR: {snr_db} dB | Seed: {seed}"
                )

    raw_df = pd.DataFrame(all_rows)

    selection_df = pd.DataFrame(all_selection_rows)

    raw_df.to_csv(
        RESULTS / "online_adaptive_raw_results.csv",
        index=False
    )

    selection_df.to_csv(
        RESULTS / "online_adaptive_selection_history.csv",
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY AND CONFIDENCE INTERVALS
    # -----------------------------------------------------

    summary = (
        raw_df.groupby(
            ["Actual Channel", "SNR (dB)", "Estimator"]
        )
        .agg(
            Mean_BER=("BER", "mean"),
            Std_BER=("BER", "std"),
            Mean_NMSE=("NMSE", "mean"),
            Mean_Selection_Accuracy=(
                "Online Selection Accuracy", "mean"
            )
        )
        .reset_index()
    )

    # t-based 95% confidence interval across five seeds
    t_value = 2.776

    margin = (
        t_value
        * summary["Std_BER"]
        / np.sqrt(len(SEEDS))
    )

    summary["BER_CI_Lower"] = np.maximum(
        0, summary["Mean_BER"] - margin
    )

    summary["BER_CI_Upper"] = np.minimum(
        1, summary["Mean_BER"] + margin
    )

    summary.to_csv(
        RESULTS / "online_adaptive_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # BER GRAPHS
    # -----------------------------------------------------

    for profile in PROFILES:

        plt.figure(figsize=(10, 6))

        subset_profile = summary[
            summary["Actual Channel"] == profile
        ]

        for estimator in ESTIMATORS:

            subset = subset_profile[
                subset_profile["Estimator"] == estimator
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
            f"Online Adaptive OFDM BER - {profile}"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS / f"online_adaptive_ber_{profile.lower()}.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # ONLINE PROFILE SELECTION ACCURACY
    # -----------------------------------------------------

    accuracy = (
        raw_df.groupby(
            ["Actual Channel", "SNR (dB)"]
        )["Online Selection Accuracy"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    for profile in PROFILES:

        subset = accuracy[
            accuracy["Actual Channel"] == profile
        ].sort_values("SNR (dB)")

        plt.plot(
            subset["SNR (dB)"],
            subset["Online Selection Accuracy"] * 100,
            marker="o",
            label=profile
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Online Profile Selection Accuracy (%)")
    plt.title("Online Channel Model Identification")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / "online_adaptive_selection_accuracy.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n---------------------------------------")
    print("ONLINE ADAPTIVE EXPERIMENT COMPLETED")
    print("---------------------------------------")

    print("Raw results:")
    print("results/online_adaptive_raw_results.csv")

    print("Selection history:")
    print("results/online_adaptive_selection_history.csv")

    print("Summary:")
    print("results/online_adaptive_summary.csv")

    print("\nSUMMARY PREVIEW:")

    print(
        summary[
            [
                "Actual Channel",
                "SNR (dB)",
                "Estimator",
                "Mean_BER",
                "Mean_NMSE",
                "Mean_Selection_Accuracy"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
