
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

# Two actual channel profiles
CHANNEL_PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for name in CHANNEL_PROFILES:
    CHANNEL_PROFILES[name] = (
        CHANNEL_PROFILES[name]
        / np.sum(CHANNEL_PROFILES[name])
    )

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ESTIMATORS = [
    "Perfect CSI",
    "LS",
    "LMMSE Matched",
    "LMMSE Mismatched"
]

# 95% t critical value for 5 independent runs (df = 4)
T_CRITICAL = 2.776


# =====================================================
# QPSK
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

def calculate_covariance(power_profile):

    power_profile = np.asarray(power_profile, dtype=float)
    power_profile = power_profile / np.sum(power_profile)

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


def generate_channel(rng, power_profile):

    taps = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    taps *= np.sqrt(power_profile)

    frequency_response = np.fft.fft(
        taps, NUM_SUBCARRIERS
    )

    return taps, frequency_response


# =====================================================
# LMMSE ESTIMATION
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

    weights = np.linalg.solve(matrix, received_pilots)

    return cross_covariance @ weights


# =====================================================
# ONE INDEPENDENT EXPERIMENT RUN
# =====================================================

def run_experiment(seed, snr_db, actual_profile_name):

    rng = np.random.default_rng(seed)

    actual_profile = CHANNEL_PROFILES[actual_profile_name]

    # The other profile is deliberately used as the
    # mismatched statistical assumption.
    other_profile_name = (
        "Uniform"
        if actual_profile_name == "Exponential"
        else "Exponential"
    )

    other_profile = CHANNEL_PROFILES[other_profile_name]

    matched_covariance = calculate_covariance(actual_profile)
    mismatched_covariance = calculate_covariance(other_profile)

    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / snr_linear

    errors = {name: 0 for name in ESTIMATORS}

    nmse_sum = {
        "LS": 0.0,
        "LMMSE Matched": 0.0,
        "LMMSE Mismatched": 0.0
    }

    total_bits = 0

    for frame in range(NUM_FRAMES):

        # Generate data
        bits = rng.integers(
            0, 2, len(DATA_INDICES) * 2, dtype=np.int8
        )

        data_symbols = qpsk_modulate(bits)

        transmitted = np.zeros(
            NUM_SUBCARRIERS, dtype=complex
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

        # Actual multipath channel
        channel_taps, true_channel = generate_channel(
            rng, actual_profile
        )

        received = np.convolve(
            transmitted_with_cp, channel_taps
        )

        # Same noise realization for all estimators
        noise = np.sqrt(noise_variance / 2) * (
            rng.normal(size=len(received))
            + 1j * rng.normal(size=len(received))
        )

        received += noise

        received_no_cp = received[
            CYCLIC_PREFIX:
            CYCLIC_PREFIX + NUM_SUBCARRIERS
        ]

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

        # LS pilot estimates and linear interpolation
        ls_pilots = received_pilots

        ls_estimate = np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOT_INDICES,
            ls_pilots.real
        ) + 1j * np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOT_INDICES,
            ls_pilots.imag
        )

        # Matched LMMSE
        matched_estimate = lmmse_estimation(
            received_pilots,
            noise_variance,
            matched_covariance
        )

        # Mismatched LMMSE
        mismatched_estimate = lmmse_estimation(
            received_pilots,
            noise_variance,
            mismatched_covariance
        )

        # Perfect CSI benchmark
        perfect_equalized = (
            received_data / true_channel[DATA_INDICES]
        )

        ls_equalized = (
            received_data / ls_estimate[DATA_INDICES]
        )

        matched_equalized = (
            received_data / matched_estimate[DATA_INDICES]
        )

        mismatched_equalized = (
            received_data / mismatched_estimate[DATA_INDICES]
        )

        detected = {
            "Perfect CSI": qpsk_demodulate(perfect_equalized),
            "LS": qpsk_demodulate(ls_equalized),
            "LMMSE Matched": qpsk_demodulate(matched_equalized),
            "LMMSE Mismatched": qpsk_demodulate(mismatched_equalized)
        }

        for estimator in ESTIMATORS:
            errors[estimator] += np.count_nonzero(
                detected[estimator] != bits
            )

        total_bits += len(bits)

        # Channel estimation NMSE
        true_power = np.mean(np.abs(true_channel) ** 2)

        nmse_sum["LS"] += (
            np.mean(np.abs(ls_estimate - true_channel) ** 2)
            / true_power
        )

        nmse_sum["LMMSE Matched"] += (
            np.mean(np.abs(matched_estimate - true_channel) ** 2)
            / true_power
        )

        nmse_sum["LMMSE Mismatched"] += (
            np.mean(np.abs(mismatched_estimate - true_channel) ** 2)
            / true_power
        )

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
            "NMSE": nmse
        })

    return rows


# =====================================================
# MAIN
# =====================================================

def main():

    print("\nOFDM ROBUSTNESS AND STATISTICAL VALIDATION")
    print("------------------------------------------")
    print(f"Independent seeds: {len(RANDOM_SEEDS)}")
    print(f"Frames per seed/SNR/profile: {NUM_FRAMES}")
    print(f"Subcarriers: {NUM_SUBCARRIERS}")
    print(f"Channel taps: {NUM_TAPS}")
    print("Channel profiles: Exponential and Uniform")
    print("Comparison: Perfect CSI, LS, matched and mismatched LMMSE")

    all_rows = []

    total_runs = (
        len(RANDOM_SEEDS)
        * len(SNR_DB_VALUES)
        * len(CHANNEL_PROFILES)
    )

    completed = 0

    for profile_name in CHANNEL_PROFILES:

        for snr_db in SNR_DB_VALUES:

            for seed in RANDOM_SEEDS:

                rows = run_experiment(
                    seed,
                    snr_db,
                    profile_name
                )

                all_rows.extend(rows)

                completed += 1

                print(
                    f"Progress: {completed}/{total_runs} | "
                    f"Profile: {profile_name} | "
                    f"SNR: {snr_db} dB | Seed: {seed}"
                )

    raw_df = pd.DataFrame(all_rows)

    raw_path = (
        RESULTS_DIR / "ofdm_robustness_raw_results.csv"
    )

    raw_df.to_csv(raw_path, index=False)

    # =================================================
    # MEAN AND 95% CONFIDENCE INTERVAL ACROSS SEEDS
    # =================================================

    summary_rows = []

    group_columns = [
        "Actual Channel",
        "SNR (dB)",
        "Estimator"
    ]

    for keys, group in raw_df.groupby(group_columns):

        profile_name, snr_db, estimator = keys

        ber_values = group["BER"].to_numpy()

        mean_ber = np.mean(ber_values)
        std_ber = np.std(ber_values, ddof=1)

        margin_ber = (
            T_CRITICAL * std_ber / np.sqrt(len(ber_values))
        )

        if estimator == "Perfect CSI":
            mean_nmse = np.nan
            lower_nmse = np.nan
            upper_nmse = np.nan
        else:
            nmse_values = group["NMSE"].to_numpy()

            mean_nmse = np.mean(nmse_values)
            std_nmse = np.std(nmse_values, ddof=1)

            margin_nmse = (
                T_CRITICAL * std_nmse
                / np.sqrt(len(nmse_values))
            )

            lower_nmse = max(0.0, mean_nmse - margin_nmse)
            upper_nmse = mean_nmse + margin_nmse

        summary_rows.append({
            "Actual Channel": profile_name,
            "SNR (dB)": snr_db,
            "Estimator": estimator,
            "Mean BER": mean_ber,
            "BER CI Lower": max(0.0, mean_ber - margin_ber),
            "BER CI Upper": min(1.0, mean_ber + margin_ber),
            "Mean NMSE": mean_nmse,
            "NMSE CI Lower": lower_nmse,
            "NMSE CI Upper": upper_nmse
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        RESULTS_DIR / "ofdm_robustness_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # =================================================
    # BER GRAPH
    # =================================================

    for profile_name in CHANNEL_PROFILES:

        plt.figure(figsize=(9, 5))

        profile_df = summary_df[
            summary_df["Actual Channel"] == profile_name
        ]

        for estimator in ESTIMATORS:

            subset = profile_df[
                profile_df["Estimator"] == estimator
            ].sort_values("SNR (dB)")

            plt.errorbar(
                subset["SNR (dB)"],
                subset["Mean BER"],
                yerr=[
                    subset["Mean BER"] - subset["BER CI Lower"],
                    subset["BER CI Upper"] - subset["Mean BER"]
                ],
                marker="o",
                capsize=4,
                label=estimator
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean BER")
        plt.title(
            f"OFDM BER Robustness - {profile_name} Channel"
        )
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"robustness_ber_{profile_name.lower()}.png",
            dpi=300
        )

        plt.close()

    # =================================================
    # NMSE GRAPH
    # =================================================

    for profile_name in CHANNEL_PROFILES:

        plt.figure(figsize=(9, 5))

        profile_df = summary_df[
            summary_df["Actual Channel"] == profile_name
        ]

        for estimator in [
            "LS",
            "LMMSE Matched",
            "LMMSE Mismatched"
        ]:

            subset = profile_df[
                profile_df["Estimator"] == estimator
            ].sort_values("SNR (dB)")

            plt.errorbar(
                subset["SNR (dB)"],
                subset["Mean NMSE"],
                yerr=[
                    subset["Mean NMSE"] - subset["NMSE CI Lower"],
                    subset["NMSE CI Upper"] - subset["Mean NMSE"]
                ],
                marker="o",
                capsize=4,
                label=estimator
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean NMSE")
        plt.title(
            f"OFDM NMSE Robustness - {profile_name} Channel"
        )
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"robustness_nmse_{profile_name.lower()}.png",
            dpi=300
        )

        plt.close()

    print("\n------------------------------------------")
    print("ROBUSTNESS VALIDATION COMPLETED")
    print(f"Raw results: {raw_path}")
    print(f"Summary results: {summary_path}")
    print("BER and NMSE graphs saved in results/")
    print("------------------------------------------")

    print("\nSUMMARY PREVIEW:")
    print(
        summary_df[
            [
                "Actual Channel",
                "SNR (dB)",
                "Estimator",
                "Mean BER",
                "BER CI Lower",
                "BER CI Upper",
                "Mean NMSE"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
