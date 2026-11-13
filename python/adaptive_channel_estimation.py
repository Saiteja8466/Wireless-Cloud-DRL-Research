
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_SEEDS = [101, 202, 303, 404, 505]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
CYCLIC_PREFIX = 16
NUM_TAPS = 8

PILOT_SPACING = 4
PILOT_INDICES = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA_INDICES = np.setdiff1d(
    np.arange(NUM_SUBCARRIERS), PILOT_INDICES
)

SNR_DB_VALUES = [0, 5, 10, 15, 20]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Candidate channel power-delay profiles
CHANNEL_PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for name in CHANNEL_PROFILES:
    CHANNEL_PROFILES[name] /= np.sum(
        CHANNEL_PROFILES[name]
    )

ESTIMATORS = [
    "Perfect CSI",
    "LS",
    "Fixed Exponential LMMSE",
    "Fixed Uniform LMMSE",
    "Adaptive LMMSE"
]


# =====================================================
# QPSK MODULATION
# =====================================================

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

    bits[:, 0] = (
        symbols.real < 0
    ).astype(np.int8)

    bits[:, 1] = (
        symbols.imag < 0
    ).astype(np.int8)

    return bits.reshape(-1)


# =====================================================
# CHANNEL COVARIANCE
# =====================================================

def calculate_covariance(power_profile):

    power_profile = np.asarray(
        power_profile,
        dtype=float
    )

    power_profile = power_profile / np.sum(
        power_profile
    )

    indices = np.arange(NUM_SUBCARRIERS)

    difference = (
        indices[:, None] - indices[None, :]
    )

    covariance = np.zeros(
        (NUM_SUBCARRIERS, NUM_SUBCARRIERS),
        dtype=complex
    )

    for tap, power in enumerate(power_profile):

        covariance += power * np.exp(
            -2j * np.pi * difference * tap
            / NUM_SUBCARRIERS
        )

    return covariance


COVARIANCES = {
    name: calculate_covariance(profile)
    for name, profile in CHANNEL_PROFILES.items()
}


# =====================================================
# CHANNEL GENERATION
# =====================================================

def generate_channel(rng, profile_name):

    powers = CHANNEL_PROFILES[profile_name]

    taps = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    taps *= np.sqrt(powers)

    frequency_response = np.fft.fft(
        taps,
        NUM_SUBCARRIERS
    )

    return taps, frequency_response


# =====================================================
# LMMSE CHANNEL ESTIMATION
# =====================================================

def lmmse_estimation(
    received_pilots,
    noise_variance,
    covariance
):

    pilot_covariance = covariance[
        np.ix_(PILOT_INDICES, PILOT_INDICES)
    ]

    cross_covariance = covariance[
        :, PILOT_INDICES
    ]

    matrix = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOT_INDICES))
    )

    weights = np.linalg.solve(
        matrix,
        received_pilots
    )

    return cross_covariance @ weights


# =====================================================
# ADAPTIVE CHANNEL PROFILE SELECTION
# =====================================================

def select_channel_profile(
    received_pilots,
    noise_variance
):

    scores = {}

    for profile_name, covariance in COVARIANCES.items():

        pilot_covariance = covariance[
            np.ix_(PILOT_INDICES, PILOT_INDICES)
        ]

        observation_covariance = (
            pilot_covariance
            + noise_variance * np.eye(len(PILOT_INDICES))
        )

        sign, log_determinant = np.linalg.slogdet(
            observation_covariance
        )

        if sign <= 0:
            raise ValueError(
                "Invalid covariance matrix"
            )

        solution = np.linalg.solve(
            observation_covariance,
            received_pilots
        )

        # Negative log-likelihood, ignoring common constants
        score = (
            log_determinant
            + np.real(
                np.vdot(received_pilots, solution)
            )
        )

        scores[profile_name] = score

    selected_profile = min(
        scores,
        key=scores.get
    )

    return selected_profile, scores


# =====================================================
# ONE INDEPENDENT SIMULATION RUN
# =====================================================

def run_experiment(
    seed,
    snr_db,
    actual_profile_name
):

    rng = np.random.default_rng(seed)

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = 1 / snr_linear

    errors = {
        name: 0 for name in ESTIMATORS
    }

    nmse_sum = {
        "LS": 0.0,
        "Fixed Exponential LMMSE": 0.0,
        "Fixed Uniform LMMSE": 0.0,
        "Adaptive LMMSE": 0.0
    }

    profile_correct = 0
    total_bits = 0

    for frame in range(NUM_FRAMES):

        # Generate data bits
        bits = rng.integers(
            0,
            2,
            len(DATA_INDICES) * 2,
            dtype=np.int8
        )

        data_symbols = qpsk_modulate(bits)

        # Construct OFDM symbol
        transmitted = np.zeros(
            NUM_SUBCARRIERS,
            dtype=complex
        )

        transmitted[PILOT_INDICES] = 1 + 0j
        transmitted[DATA_INDICES] = data_symbols

        # OFDM modulation
        time_signal = (
            np.fft.ifft(transmitted)
            * np.sqrt(NUM_SUBCARRIERS)
        )

        transmitted_with_cp = np.concatenate([
            time_signal[-CYCLIC_PREFIX:],
            time_signal
        ])

        # Generate actual channel
        channel_taps, true_channel = generate_channel(
            rng,
            actual_profile_name
        )

        # Multipath convolution
        received = np.convolve(
            transmitted_with_cp,
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
            CYCLIC_PREFIX:
            CYCLIC_PREFIX + NUM_SUBCARRIERS
        ]

        # OFDM demodulation
        received_frequency = (
            np.fft.fft(received_no_cp)
            / np.sqrt(NUM_SUBCARRIERS)
        )

        received_pilots = (
            received_frequency[PILOT_INDICES]
        )

        received_data = (
            received_frequency[DATA_INDICES]
        )

        # -------------------------------------------------
        # LS ESTIMATION
        # -------------------------------------------------

        ls_estimate = np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOT_INDICES,
            received_pilots.real
        ) + 1j * np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOT_INDICES,
            received_pilots.imag
        )

        # -------------------------------------------------
        # FIXED EXPONENTIAL LMMSE
        # -------------------------------------------------

        exponential_estimate = lmmse_estimation(
            received_pilots,
            noise_variance,
            COVARIANCES["Exponential"]
        )

        # -------------------------------------------------
        # FIXED UNIFORM LMMSE
        # -------------------------------------------------

        uniform_estimate = lmmse_estimation(
            received_pilots,
            noise_variance,
            COVARIANCES["Uniform"]
        )

        # -------------------------------------------------
        # ADAPTIVE LMMSE
        # -------------------------------------------------

        selected_profile, scores = select_channel_profile(
            received_pilots,
            noise_variance
        )

        adaptive_estimate = lmmse_estimation(
            received_pilots,
            noise_variance,
            COVARIANCES[selected_profile]
        )

        if selected_profile == actual_profile_name:
            profile_correct += 1

        # -------------------------------------------------
        # EQUALIZATION
        # -------------------------------------------------

        true_data_channel = (
            true_channel[DATA_INDICES]
        )

        perfect_equalized = (
            received_data / true_data_channel
        )

        ls_equalized = (
            received_data / ls_estimate[DATA_INDICES]
        )

        exponential_equalized = (
            received_data
            / exponential_estimate[DATA_INDICES]
        )

        uniform_equalized = (
            received_data
            / uniform_estimate[DATA_INDICES]
        )

        adaptive_equalized = (
            received_data
            / adaptive_estimate[DATA_INDICES]
        )

        # -------------------------------------------------
        # DEMODULATION
        # -------------------------------------------------

        detected = {
            "Perfect CSI": qpsk_demodulate(
                perfect_equalized
            ),

            "LS": qpsk_demodulate(
                ls_equalized
            ),

            "Fixed Exponential LMMSE": qpsk_demodulate(
                exponential_equalized
            ),

            "Fixed Uniform LMMSE": qpsk_demodulate(
                uniform_equalized
            ),

            "Adaptive LMMSE": qpsk_demodulate(
                adaptive_equalized
            )
        }

        for estimator in ESTIMATORS:

            errors[estimator] += np.count_nonzero(
                detected[estimator] != bits
            )

        total_bits += len(bits)

        # -------------------------------------------------
        # NMSE
        # -------------------------------------------------

        true_power = np.mean(
            np.abs(true_channel) ** 2
        )

        estimates = {
            "LS": ls_estimate,
            "Fixed Exponential LMMSE": exponential_estimate,
            "Fixed Uniform LMMSE": uniform_estimate,
            "Adaptive LMMSE": adaptive_estimate
        }

        for estimator, estimate in estimates.items():

            nmse_sum[estimator] += (
                np.mean(
                    np.abs(estimate - true_channel) ** 2
                ) / true_power
            )

    # -----------------------------------------------------
    # SAVE RUN RESULTS
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
            "Actual Channel": actual_profile_name,
            "SNR (dB)": snr_db,
            "Estimator": estimator,
            "BER": ber,
            "NMSE": nmse,
            "Profile Selection Accuracy": (
                profile_correct / NUM_FRAMES
            )
        })

    return rows


# =====================================================
# MAIN EXPERIMENT
# =====================================================

def main():

    all_rows = []

    total_runs = (
        len(RANDOM_SEEDS)
        * len(SNR_DB_VALUES)
        * len(CHANNEL_PROFILES)
    )

    completed = 0

    print("\nADAPTIVE OFDM CHANNEL ESTIMATION")
    print("--------------------------------")
    print(f"Independent seeds: {len(RANDOM_SEEDS)}")
    print(f"Frames per run: {NUM_FRAMES}")
    print(f"Subcarriers: {NUM_SUBCARRIERS}")
    print(f"Channel profiles: {list(CHANNEL_PROFILES.keys())}")
    print("Adaptive selection: Pilot-based likelihood")

    for actual_profile in CHANNEL_PROFILES:

        for snr_db in SNR_DB_VALUES:

            for seed in RANDOM_SEEDS:

                rows = run_experiment(
                    seed,
                    snr_db,
                    actual_profile
                )

                all_rows.extend(rows)

                completed += 1

                print(
                    f"Progress: {completed}/{total_runs} | "
                    f"Channel: {actual_profile} | "
                    f"SNR: {snr_db} dB | "
                    f"Seed: {seed}"
                )

    raw_df = pd.DataFrame(all_rows)

    raw_path = (
        RESULTS_DIR / "adaptive_channel_raw_results.csv"
    )

    raw_df.to_csv(raw_path, index=False)

    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    summary = (
        raw_df.groupby(
            ["Actual Channel", "SNR (dB)", "Estimator"]
        )
        .agg(
            Mean_BER=("BER", "mean"),
            Std_BER=("BER", "std"),
            Mean_NMSE=("NMSE", "mean"),
            Std_NMSE=("NMSE", "std"),
            Mean_Profile_Accuracy=(
                "Profile Selection Accuracy", "mean"
            )
        )
        .reset_index()
    )

    # 95% t interval across five independent seeds
    t_critical = 2.776
    standard_error = (
        summary["Std_BER"] / np.sqrt(len(RANDOM_SEEDS))
    )

    summary["BER_CI_Lower"] = np.maximum(
        0,
        summary["Mean_BER"] - t_critical * standard_error
    )

    summary["BER_CI_Upper"] = np.minimum(
        1,
        summary["Mean_BER"] + t_critical * standard_error
    )

    summary_path = (
        RESULTS_DIR / "adaptive_channel_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    # -----------------------------------------------------
    # BER GRAPHS
    # -----------------------------------------------------

    for profile in CHANNEL_PROFILES:

        plt.figure(figsize=(10, 6))

        profile_data = summary[
            summary["Actual Channel"] == profile
        ]

        for estimator in ESTIMATORS:

            subset = profile_data[
                profile_data["Estimator"] == estimator
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
            f"Adaptive OFDM Channel Estimation - {profile}"
        )

        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            / f"adaptive_ber_{profile.lower()}.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # PROFILE SELECTION ACCURACY
    # -----------------------------------------------------

    plt.figure(figsize=(9, 5))

    for profile in CHANNEL_PROFILES:

        subset = summary[
            (summary["Actual Channel"] == profile)
            & (summary["Estimator"] == "Adaptive LMMSE")
        ].sort_values("SNR (dB)")

        plt.plot(
            subset["SNR (dB)"],
            subset["Mean_Profile_Accuracy"] * 100,
            marker="o",
            label=profile
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Correct Profile Selection (%)")
    plt.title("Adaptive Channel Profile Identification")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "adaptive_profile_accuracy.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------

    print("\n--------------------------------")
    print("ADAPTIVE CHANNEL EXPERIMENT DONE")
    print("--------------------------------")

    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")

    print("\nSUMMARY PREVIEW:")

    print(
        summary[
            [
                "Actual Channel",
                "SNR (dB)",
                "Estimator",
                "Mean_BER",
                "Mean_NMSE",
                "Mean_Profile_Accuracy"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
