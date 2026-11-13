
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_SEED = 20261113
NUM_PACKETS = 5000
PACKET_BITS = 1024
MAX_RETRANSMISSIONS = 3
MAX_ATTEMPTS = MAX_RETRANSMISSIONS + 1

SNR_DB_VALUES = [0, 5, 10, 15, 20]
ESTIMATORS = ["Perfect CSI", "LS", "LMMSE"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# QPSK modulation
# --------------------------------------------------

def qpsk_modulate(bits):
    bit_pairs = bits.reshape(-1, 2)

    symbols = (
        (1 - 2 * bit_pairs[:, 0])
        + 1j * (1 - 2 * bit_pairs[:, 1])
    ) / np.sqrt(2)

    return symbols


def qpsk_demodulate(symbols):
    bits = np.empty((len(symbols), 2), dtype=np.int8)

    bits[:, 0] = (symbols.real < 0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0).astype(np.int8)

    return bits.reshape(-1)


# --------------------------------------------------
# Wilson confidence interval
# --------------------------------------------------

def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1 + (z ** 2 / total)

    centre = (p + z ** 2 / (2 * total)) / denominator

    margin = (
        z
        * np.sqrt(
            (p * (1 - p) / total)
            + (z ** 2 / (4 * total ** 2))
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


# --------------------------------------------------
# One estimator's ARQ experiment
# All estimators use the same generated random samples.
# --------------------------------------------------

def simulate_estimator(snr_db, snr_index, estimator):

    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / snr_linear

    packet_errors_first_attempt = 0
    delivered_packets = 0
    total_attempts = 0
    total_channel_uses = 0

    for packet_index in range(NUM_PACKETS):

        # Same packet data for every estimator
        bit_rng = np.random.default_rng(
            np.random.SeedSequence(
                [BASE_SEED, snr_index, packet_index, 999]
            )
        )

        bits = bit_rng.integers(
            0, 2, PACKET_BITS, dtype=np.int8
        )

        transmitted_symbols = qpsk_modulate(bits)

        packet_delivered = False

        for attempt in range(MAX_ATTEMPTS):

            # Identical channel and noise for all estimators
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [BASE_SEED, snr_index, packet_index, attempt]
                )
            )

            h = (
                rng.normal() + 1j * rng.normal()
            ) / np.sqrt(2)

            pilot_noise = np.sqrt(noise_variance / 2) * (
                rng.normal() + 1j * rng.normal()
            )

            data_noise = np.sqrt(noise_variance / 2) * (
                rng.normal(size=len(transmitted_symbols))
                + 1j * rng.normal(size=len(transmitted_symbols))
            )

            # Known pilot symbol = 1
            received_pilot = h + pilot_noise

            received_data = (
                h * transmitted_symbols + data_noise
            )

            # Channel estimation
            if estimator == "Perfect CSI":
                h_estimated = h

            elif estimator == "LS":
                h_estimated = received_pilot

            elif estimator == "LMMSE":
                h_estimated = (
                    received_pilot / (1 + noise_variance)
                )

            else:
                raise ValueError("Unknown channel estimator")

            # Equalization
            equalized_symbols = received_data / h_estimated

            detected_bits = qpsk_demodulate(equalized_symbols)

            packet_error = np.any(detected_bits != bits)

            if attempt == 0 and packet_error:
                packet_errors_first_attempt += 1

            total_attempts += 1

            # One pilot plus one data symbol per QPSK symbol
            total_channel_uses += 1 + len(transmitted_symbols)

            if not packet_error:
                packet_delivered = True
                break

        if packet_delivered:
            delivered_packets += 1

    first_attempt_per = (
        packet_errors_first_attempt / NUM_PACKETS
    )

    delivery_ratio = delivered_packets / NUM_PACKETS

    average_attempts = total_attempts / NUM_PACKETS

    # Each QPSK symbol carries two information bits.
    goodput = (
        delivered_packets * PACKET_BITS / total_channel_uses
    )

    per_ci_low, per_ci_high = wilson_interval(
        packet_errors_first_attempt, NUM_PACKETS
    )

    delivery_ci_low, delivery_ci_high = wilson_interval(
        delivered_packets, NUM_PACKETS
    )

    return {
        "SNR (dB)": snr_db,
        "Estimator": estimator,
        "First Attempt PER": first_attempt_per,
        "PER CI Lower": per_ci_low,
        "PER CI Upper": per_ci_high,
        "Delivery Ratio": delivery_ratio,
        "Delivery CI Lower": delivery_ci_low,
        "Delivery CI Upper": delivery_ci_high,
        "Average Attempts": average_attempts,
        "Goodput (bits/channel use)": goodput,
    }


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    all_results = []

    print("\nFAIR CHANNEL ESTIMATION AND ARQ VALIDATION")
    print("------------------------------------------")
    print(f"Packets per SNR: {NUM_PACKETS}")
    print(f"Packet size: {PACKET_BITS} bits")
    print(f"Maximum attempts: {MAX_ATTEMPTS}")
    print("Common random numbers: ENABLED\n")

    for snr_index, snr_db in enumerate(SNR_DB_VALUES):

        print(f"\nSNR = {snr_db} dB")

        for estimator in ESTIMATORS:

            result = simulate_estimator(
                snr_db, snr_index, estimator
            )

            all_results.append(result)

            print(
                f"{estimator:12s} | "
                f"PER: {result['First Attempt PER']:.4f} | "
                f"Delivery: {result['Delivery Ratio'] * 100:.2f}% | "
                f"Attempts: {result['Average Attempts']:.3f} | "
                f"Goodput: {result['Goodput (bits/channel use)']:.4f}"
            )

    df = pd.DataFrame(all_results)

    csv_path = RESULTS_DIR / "fair_channel_validation.csv"
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------
    # PER graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for estimator in ESTIMATORS:
        subset = df[df["Estimator"] == estimator]

        plt.plot(
            subset["SNR (dB)"],
            subset["First Attempt PER"],
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("First Attempt Packet Error Rate")
    plt.title("Fair Channel Estimation: Packet Error Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fair_channel_per.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # Delivery ratio graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for estimator in ESTIMATORS:
        subset = df[df["Estimator"] == estimator]

        plt.plot(
            subset["SNR (dB)"],
            subset["Delivery Ratio"] * 100,
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Packet Delivery Ratio (%)")
    plt.title("Fair Channel Estimation: ARQ Delivery")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fair_channel_delivery.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------------------
    # Goodput graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    for estimator in ESTIMATORS:
        subset = df[df["Estimator"] == estimator]

        plt.plot(
            subset["SNR (dB)"],
            subset["Goodput (bits/channel use)"],
            marker="o",
            label=estimator
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Goodput (bits/channel use)")
    plt.title("Fair Channel Estimation: Goodput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fair_channel_goodput.png",
        dpi=300
    )
    plt.close()

    print("\n------------------------------------------")
    print("FAIR VALIDATION COMPLETED")
    print(f"CSV saved: {csv_path}")
    print("Graphs saved in results/")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
