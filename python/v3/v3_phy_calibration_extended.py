import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# V3 PHY CALIBRATION: OFDM + CHANNEL ESTIMATION + MODULATION
# ============================================================

SEED = 20261114
NUM_SUBCARRIERS = 64
CYCLIC_PREFIX = 16
NUM_TAPS = 8
NUM_FRAMES = 1000

SNR_DB_VALUES = list(range(0, 31, 2))
MODULATIONS = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
ESTIMATORS = ["Perfect CSI", "LS", "LMMSE"]

PILOT_SPACING = 4
PILOT_INDICES = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA_INDICES = np.setdiff1d(
    np.arange(NUM_SUBCARRIERS), PILOT_INDICES
)

NUM_PILOTS = len(PILOT_INDICES)
NUM_DATA = len(DATA_INDICES)

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "v3"
GRAPHS_DIR = BASE_DIR / "graphs" / "v3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# MODULATION
# ============================================================

def bits_per_symbol(modulation):
    return {
        "BPSK": 1,
        "QPSK": 2,
        "16-QAM": 4,
        "64-QAM": 6
    }[modulation]


def modulate(bits, modulation):

    if modulation == "BPSK":
        return (1 - 2 * bits).astype(complex)

    k = bits_per_symbol(modulation)
    M = 2 ** k
    side = int(np.sqrt(M))

    groups = bits.reshape(-1, k)
    weights = 2 ** np.arange(k - 1, -1, -1)
    indices = groups @ weights

    gray = np.arange(side) ^ (np.arange(side) >> 1)
    levels = 2 * np.arange(side) - (side - 1)

    i_gray = indices >> (k // 2)
    q_gray = indices & (side - 1)

    i_level = levels[gray[i_gray]]
    q_level = levels[gray[q_gray]]

    scale = np.sqrt(2 * (M - 1) / 3)

    return (i_level + 1j * q_level) / scale


def demodulate(symbols, modulation):

    if modulation == "BPSK":
        return (symbols.real < 0).astype(np.int8)

    k = bits_per_symbol(modulation)
    M = 2 ** k
    side = int(np.sqrt(M))

    gray = np.arange(side) ^ (np.arange(side) >> 1)
    levels = 2 * np.arange(side) - (side - 1)

    scale = np.sqrt(2 * (M - 1) / 3)
    received = symbols * scale

    i_index = np.argmin(
        np.abs(received.real[:, None] - levels), axis=1
    )
    q_index = np.argmin(
        np.abs(received.imag[:, None] - levels), axis=1
    )

    inverse_gray = np.argsort(gray)

    i_gray = inverse_gray[i_index]
    q_gray = inverse_gray[q_index]

    indices = (i_gray << (k // 2)) | q_gray

    weights = 2 ** np.arange(k - 1, -1, -1)

    return (
        (indices[:, None] & weights) > 0
    ).astype(np.int8).ravel()


# ============================================================
# RAYLEIGH MULTIPATH CHANNEL
# ============================================================

def generate_channel():

    powers = np.exp(-np.arange(NUM_TAPS))
    powers = powers / np.sum(powers)

    channel = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    channel *= np.sqrt(powers)

    frequency_response = np.fft.fft(
        channel, NUM_SUBCARRIERS
    )

    return channel, frequency_response


# ============================================================
# CHANNEL COVARIANCE AND LMMSE
# ============================================================

def calculate_covariance():

    powers = np.exp(-np.arange(NUM_TAPS))
    powers /= np.sum(powers)

    indices = np.arange(NUM_SUBCARRIERS)

    difference = indices[:, None] - indices[None, :]

    covariance = np.zeros(
        (NUM_SUBCARRIERS, NUM_SUBCARRIERS),
        dtype=complex
    )

    for tap, power in enumerate(powers):
        covariance += power * np.exp(
            -2j * np.pi * difference * tap
            / NUM_SUBCARRIERS
        )

    return covariance


CHANNEL_COVARIANCE = calculate_covariance()


def lmmse_estimation(pilot_received, noise_variance):

    pilot_covariance = CHANNEL_COVARIANCE[
        np.ix_(PILOT_INDICES, PILOT_INDICES)
    ]

    cross_covariance = CHANNEL_COVARIANCE[
        :, PILOT_INDICES
    ]

    regularized = (
        pilot_covariance
        + noise_variance * np.eye(NUM_PILOTS)
    )

    weights = np.linalg.solve(
        regularized, pilot_received
    )

    return cross_covariance @ weights


# ============================================================
# LS ESTIMATION
# ============================================================

def ls_estimation(pilot_received):

    all_indices = np.arange(NUM_SUBCARRIERS)

    real_part = np.interp(
        all_indices,
        PILOT_INDICES,
        pilot_received.real
    )

    imag_part = np.interp(
        all_indices,
        PILOT_INDICES,
        pilot_received.imag
    )

    return real_part + 1j * imag_part


# ============================================================
# MAIN CALIBRATION
# ============================================================

def run_calibration():

    records = []

    print("\nV3 PHY CALIBRATION")
    print("==================")
    print(f"Subcarriers: {NUM_SUBCARRIERS}")
    print(f"Pilots: {NUM_PILOTS}")
    print(f"Data subcarriers: {NUM_DATA}")
    print(f"Channel taps: {NUM_TAPS}")
    print(f"Frames per combination: {NUM_FRAMES}")
    print(f"Modulations: {MODULATIONS}")
    print(f"Estimators: {ESTIMATORS}")

    for modulation in MODULATIONS:

        k = bits_per_symbol(modulation)

        print(f"\n--- {modulation} ---")

        for snr_db in SNR_DB_VALUES:

            snr_linear = 10 ** (snr_db / 10)
            noise_variance = 1 / snr_linear

            error_counts = {
                estimator: 0 for estimator in ESTIMATORS
            }

            nmse_sums = {
                "LS": 0.0,
                "LMMSE": 0.0
            }

            total_bits = 0

            for frame in range(NUM_FRAMES):

                # Generate the same transmitted data
                # for all three estimator comparisons.
                bits = rng.integers(
                    0, 2, NUM_DATA * k, dtype=np.int8
                )

                data_symbols = modulate(bits, modulation)

                transmitted = np.zeros(
                    NUM_SUBCARRIERS, dtype=complex
                )

                transmitted[PILOT_INDICES] = 1 + 0j
                transmitted[DATA_INDICES] = data_symbols

                # OFDM modulation and cyclic prefix
                time_signal = (
                    np.fft.ifft(transmitted)
                    * np.sqrt(NUM_SUBCARRIERS)
                )

                transmitted_cp = np.concatenate([
                    time_signal[-CYCLIC_PREFIX:],
                    time_signal
                ])

                # Multipath channel
                channel, true_channel = generate_channel()

                received = np.convolve(
                    transmitted_cp, channel
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

                # Pilot observations
                received_pilots = (
                    received_frequency[PILOT_INDICES]
                )

                # Channel estimates
                ls_estimate = ls_estimation(
                    received_pilots
                )

                lmmse_estimate = lmmse_estimation(
                    received_pilots, noise_variance
                )

                estimates = {
                    "Perfect CSI": true_channel,
                    "LS": ls_estimate,
                    "LMMSE": lmmse_estimate
                }

                received_data = (
                    received_frequency[DATA_INDICES]
                )

                for estimator, estimate in estimates.items():

                    data_estimate = estimate[DATA_INDICES]

                    # Protect against division by zero.
                    magnitude = np.maximum(
                        np.abs(data_estimate), 1e-10
                    )

                    safe_estimate = (
                        magnitude
                        * np.exp(1j * np.angle(data_estimate))
                    )

                    equalized = (
                        received_data / safe_estimate
                    )

                    detected_bits = demodulate(
                        equalized, modulation
                    )

                    error_counts[estimator] += np.count_nonzero(
                        detected_bits != bits
                    )

                # Channel estimation NMSE
                channel_power = np.mean(
                    np.abs(true_channel) ** 2
                )

                nmse_sums["LS"] += (
                    np.mean(np.abs(ls_estimate - true_channel) ** 2)
                    / channel_power
                )

                nmse_sums["LMMSE"] += (
                    np.mean(
                        np.abs(lmmse_estimate - true_channel) ** 2
                    ) / channel_power
                )

                total_bits += len(bits)

            for estimator in ESTIMATORS:

                ber = error_counts[estimator] / total_bits

                if estimator == "Perfect CSI":
                    nmse = np.nan
                else:
                    nmse = (
                        nmse_sums[estimator] / NUM_FRAMES
                    )

                records.append({
                    "Modulation": modulation,
                    "Bits per Symbol": k,
                    "SNR (dB)": snr_db,
                    "Estimator": estimator,
                    "BER": ber,
                    "NMSE": nmse,
                    "Frames": NUM_FRAMES,
                    "Total Bits": total_bits
                })

                print(
                    f"SNR: {snr_db:2d} dB | "
                    f"{estimator:12s} | "
                    f"BER: {ber:.6e} | "
                    f"NMSE: {nmse:.6e}"
                )

    return pd.DataFrame(records)


# ============================================================
# SAVE RESULTS AND GRAPHS
# ============================================================

def save_results(df):

    csv_path = RESULTS_DIR / "v3_phy_calibration_extended.csv"
    df.to_csv(csv_path, index=False)

    # BER graph for each modulation
    for modulation in MODULATIONS:

        subset_mod = df[
            df["Modulation"] == modulation
        ]

        plt.figure(figsize=(8, 5))

        for estimator in ESTIMATORS:

            subset = subset_mod[
                subset_mod["Estimator"] == estimator
            ]

            plt.semilogy(
                subset["SNR (dB)"],
                subset["BER"],
                marker="o",
                label=estimator
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate")
        plt.title(
            f"V3 OFDM BER Performance - {modulation}"
        )
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            GRAPHS_DIR / f"v3_phy_ber_extended_{modulation.lower().replace('-', '')}.png",
            dpi=300
        )

        plt.close()

    # NMSE graph
    plt.figure(figsize=(8, 5))

    for estimator in ["LS", "LMMSE"]:

        subset = df[
            (df["Estimator"] == estimator)
            & (df["Modulation"] == "QPSK")
        ]

        plt.semilogy(
            subset["SNR (dB)"],
            subset["NMSE"],
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Normalized Mean Square Error")
    plt.title("V3 OFDM Channel Estimation NMSE - QPSK")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "v3_phy_nmse_extended.png",
        dpi=300
    )

    plt.close()

    print("\nV3 PHY CALIBRATION COMPLETED")
    print(f"CSV: {csv_path}")
    print(f"Graphs: {GRAPHS_DIR}")


if __name__ == "__main__":

    results_df = run_calibration()
    save_results(results_df)
