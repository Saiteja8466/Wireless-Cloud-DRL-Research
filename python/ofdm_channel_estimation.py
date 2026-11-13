
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

SEED = 20261113
NUM_SUBCARRIERS = 64
CYCLIC_PREFIX = 16
NUM_TAPS = 8
NUM_FRAMES = 3000

SNR_DB_VALUES = [0, 5, 10, 15, 20]

# Comb pilots: one pilot every four subcarriers
PILOT_SPACING = 4
PILOT_INDICES = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA_INDICES = np.setdiff1d(
    np.arange(NUM_SUBCARRIERS), PILOT_INDICES
)

NUM_PILOTS = len(PILOT_INDICES)
NUM_DATA = len(DATA_INDICES)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)


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
    bits = np.empty((len(symbols), 2), dtype=np.int8)

    bits[:, 0] = (symbols.real < 0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0).astype(np.int8)

    return bits.reshape(-1)


# =====================================================
# CHANNEL GENERATION
# =====================================================

def generate_channel():
    # Exponentially decaying power delay profile
    powers = np.exp(-np.arange(NUM_TAPS))
    powers = powers / np.sum(powers)

    channel = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    channel = channel * np.sqrt(powers)

    # Frequency response of the multipath channel
    channel_frequency = np.fft.fft(
        channel, NUM_SUBCARRIERS
    )

    return channel, channel_frequency


# =====================================================
# LMMSE FREQUENCY-DOMAIN ESTIMATION
# =====================================================

def lmmse_estimation(pilot_received, noise_variance, covariance):

    pilot_covariance = covariance[
        np.ix_(PILOT_INDICES, PILOT_INDICES)
    ]

    cross_covariance = covariance[
        :, PILOT_INDICES
    ]

    regularized_matrix = (
        pilot_covariance
        + noise_variance * np.eye(NUM_PILOTS)
    )

    weights = np.linalg.solve(
        regularized_matrix,
        pilot_received
    )

    return cross_covariance @ weights


# =====================================================
# CHANNEL CORRELATION MODEL
# =====================================================

def calculate_covariance():

    powers = np.exp(-np.arange(NUM_TAPS))
    powers = powers / np.sum(powers)

    delay_indices = np.arange(NUM_TAPS)

    frequency_indices = np.arange(NUM_SUBCARRIERS)

    difference = (
        frequency_indices[:, None]
        - frequency_indices[None, :]
    )

    covariance = np.zeros(
        (NUM_SUBCARRIERS, NUM_SUBCARRIERS),
        dtype=complex
    )

    for tap, power in zip(delay_indices, powers):

        covariance += power * np.exp(
            -2j * np.pi * difference * tap
            / NUM_SUBCARRIERS
        )

    return covariance


CHANNEL_COVARIANCE = calculate_covariance()


# =====================================================
# MAIN SIMULATION
# =====================================================

def run_simulation():

    results = []

    print("\nOFDM CHANNEL ESTIMATION SIMULATION")
    print("----------------------------------")
    print(f"Subcarriers: {NUM_SUBCARRIERS}")
    print(f"Pilot subcarriers: {NUM_PILOTS}")
    print(f"Data subcarriers: {NUM_DATA}")
    print(f"Channel taps: {NUM_TAPS}")
    print(f"Frames per SNR: {NUM_FRAMES}")

    for snr_db in SNR_DB_VALUES:

        snr_linear = 10 ** (snr_db / 10)

        noise_variance = 1 / snr_linear

        errors = {
            "Perfect CSI": 0,
            "LS": 0,
            "LMMSE": 0
        }

        total_bits = 0

        nmse = {
            "LS": 0.0,
            "LMMSE": 0.0
        }

        for frame in range(NUM_FRAMES):

            # Generate QPSK data
            bits = rng.integers(
                0, 2, NUM_DATA * 2, dtype=np.int8
            )

            data_symbols = qpsk_modulate(bits)

            # Build OFDM frequency-domain symbol
            transmitted = np.zeros(
                NUM_SUBCARRIERS, dtype=complex
            )

            transmitted[PILOT_INDICES] = 1 + 0j
            transmitted[DATA_INDICES] = data_symbols

            # OFDM modulation
            time_signal = np.fft.ifft(
                transmitted
            ) * np.sqrt(NUM_SUBCARRIERS)

            # Add cyclic prefix
            transmitted_with_cp = np.concatenate([
                time_signal[-CYCLIC_PREFIX:],
                time_signal
            ])

            # Generate multipath channel
            channel, true_channel = generate_channel()

            # Convolution with channel
            received = np.convolve(
                transmitted_with_cp, channel
            )

            # AWGN
            noise = np.sqrt(noise_variance / 2) * (
                rng.normal(size=len(received))
                + 1j * rng.normal(size=len(received))
            )

            received += noise

            # Remove CP
            received_no_cp = received[
                CYCLIC_PREFIX:
                CYCLIC_PREFIX + NUM_SUBCARRIERS
            ]

            # OFDM demodulation
            received_frequency = (
                np.fft.fft(received_no_cp)
                / np.sqrt(NUM_SUBCARRIERS)
            )

            # Extract pilots
            received_pilots = (
                received_frequency[PILOT_INDICES]
            )

            # LS estimation at pilot locations
            ls_pilot_estimate = received_pilots

            # Interpolate LS estimates across subcarriers
            ls_estimate = np.interp(
                np.arange(NUM_SUBCARRIERS),
                PILOT_INDICES,
                ls_pilot_estimate.real
            ) + 1j * np.interp(
                np.arange(NUM_SUBCARRIERS),
                PILOT_INDICES,
                ls_pilot_estimate.imag
            )

            # LMMSE estimation
            lmmse_estimate = lmmse_estimation(
                received_pilots,
                noise_variance,
                CHANNEL_COVARIANCE
            )

            # Received data subcarriers
            received_data = (
                received_frequency[DATA_INDICES]
            )

            true_data_channel = (
                true_channel[DATA_INDICES]
            )

            # Equalization
            perfect_equalized = (
                received_data / true_data_channel
            )

            ls_equalized = (
                received_data / ls_estimate[DATA_INDICES]
            )

            lmmse_equalized = (
                received_data
                / lmmse_estimate[DATA_INDICES]
            )

            # Detection
            perfect_bits = qpsk_demodulate(
                perfect_equalized
            )

            ls_bits = qpsk_demodulate(
                ls_equalized
            )

            lmmse_bits = qpsk_demodulate(
                lmmse_equalized
            )

            errors["Perfect CSI"] += np.count_nonzero(
                perfect_bits != bits
            )

            errors["LS"] += np.count_nonzero(
                ls_bits != bits
            )

            errors["LMMSE"] += np.count_nonzero(
                lmmse_bits != bits
            )

            total_bits += len(bits)

            # Channel estimation NMSE
            nmse["LS"] += np.mean(
                np.abs(ls_estimate - true_channel) ** 2
            ) / np.mean(np.abs(true_channel) ** 2)

            nmse["LMMSE"] += np.mean(
                np.abs(lmmse_estimate - true_channel) ** 2
            ) / np.mean(np.abs(true_channel) ** 2)

        # Calculate BER
        for estimator in errors:

            ber = errors[estimator] / total_bits

            if estimator == "Perfect CSI":
                estimation_nmse = np.nan
            else:
                estimation_nmse = (
                    nmse[estimator] / NUM_FRAMES
                )

            results.append({
                "SNR (dB)": snr_db,
                "Estimator": estimator,
                "BER": ber,
                "NMSE": estimation_nmse
            })

            print(
                f"SNR: {snr_db:2d} dB | "
                f"{estimator:12s} | "
                f"BER: {ber:.6e} | "
                f"NMSE: {estimation_nmse:.6e}"
            )

    return pd.DataFrame(results)


# =====================================================
# SAVE RESULTS AND GRAPHS
# =====================================================

def save_results(df):

    csv_path = (
        RESULTS_DIR / "ofdm_channel_estimation.csv"
    )

    df.to_csv(csv_path, index=False)

    # BER graph
    plt.figure(figsize=(8, 5))

    for estimator in [
        "Perfect CSI", "LS", "LMMSE"
    ]:

        subset = df[
            df["Estimator"] == estimator
        ]

        plt.semilogy(
            subset["SNR (dB)"],
            subset["BER"],
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("OFDM Channel Estimation BER")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "ofdm_channel_estimation_ber.png",
        dpi=300
    )

    plt.close()

    # NMSE graph
    plt.figure(figsize=(8, 5))

    for estimator in ["LS", "LMMSE"]:

        subset = df[
            df["Estimator"] == estimator
        ]

        plt.semilogy(
            subset["SNR (dB)"],
            subset["NMSE"],
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Normalized Mean Square Error")
    plt.title("OFDM Channel Estimation NMSE")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "ofdm_channel_estimation_nmse.png",
        dpi=300
    )

    plt.close()

    print("\nResults saved successfully:")
    print(csv_path)
    print("results/ofdm_channel_estimation_ber.png")
    print("results/ofdm_channel_estimation_nmse.png")


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    dataframe = run_simulation()

    save_results(dataframe)

    print("\nOFDM CHANNEL ESTIMATION COMPLETED")
