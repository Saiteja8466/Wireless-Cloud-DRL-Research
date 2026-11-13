
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the existing, validated OFDM implementation
import v3_phy_calibration_extended as phy


# ============================================================
# V3 PACKET-LEVEL PHY CALIBRATION
# ============================================================

PACKET_BYTES = 1500
PACKET_BITS = PACKET_BYTES * 8

SNR_VALUES = [16, 20, 26, 30]
MODULATIONS = ["BPSK", "QPSK", "16-QAM", "64-QAM"]

# Initial experiment: five seeds, 50 packets per condition.
# We will increase this after checking the first results.
SEEDS = [7, 42, 99, 123, 2026]
PACKETS_PER_SEED = 200

RESULTS_DIR = Path("results/v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_INDICES = phy.DATA_INDICES
PILOT_INDICES = phy.PILOT_INDICES

NUM_DATA = len(DATA_INDICES)
NUM_SUBCARRIERS = phy.NUM_SUBCARRIERS
CP_LENGTH = phy.CYCLIC_PREFIX


def transmit_one_packet(snr_db, modulation):
    """
    Transmit one uncoded 1500-byte packet.

    The Rayleigh channel remains constant throughout this packet.
    AWGN is independently generated for each OFDM symbol.
    LMMSE channel estimation is performed on every OFDM symbol.
    """

    k = phy.bits_per_symbol(modulation)
    bits = phy.rng.integers(
        0, 2, PACKET_BITS, dtype=np.int8
    )

    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1.0 / snr_linear

    # One block-fading channel realization per packet
    channel, true_channel = phy.generate_channel()

    total_errors = 0
    bit_position = 0

    bits_per_ofdm_symbol = NUM_DATA * k
    number_of_symbols = (
        PACKET_BITS + bits_per_ofdm_symbol - 1
    ) // bits_per_ofdm_symbol

    while bit_position < PACKET_BITS:

        remaining = PACKET_BITS - bit_position
        valid_bits = min(remaining, bits_per_ofdm_symbol)

        symbol_bits = bits[
            bit_position:bit_position + valid_bits
        ]

        # Padding is used only to complete the final OFDM symbol.
        ppadding = bits_per_ofdm_symbol - valid_bits

        if ppadding > 0:
           padded_bits = np.concatenate([
           symbol_bits,
           np.zeros(ppadding, dtype=np.int8)
        ])
        else:
           padded_bits = symbol_bits

        data_symbols = phy.modulate(
            padded_bits, modulation
        )

        transmitted = np.zeros(
            NUM_SUBCARRIERS, dtype=complex
        )

        transmitted[PILOT_INDICES] = 1.0 + 0j
        transmitted[DATA_INDICES] = data_symbols

        # OFDM modulation
        time_signal = (
            np.fft.ifft(transmitted)
            * np.sqrt(NUM_SUBCARRIERS)
        )

        transmitted_cp = np.concatenate([
            time_signal[-CP_LENGTH:],
            time_signal
        ])

        # Same channel throughout this packet
        received = np.convolve(
            transmitted_cp, channel
        )

        # Independent AWGN for each OFDM symbol
        noise = np.sqrt(noise_variance / 2) * (
            phy.rng.normal(size=len(received))
            + 1j * phy.rng.normal(size=len(received))
        )

        received += noise

        # Remove cyclic prefix and demodulate
        received_no_cp = received[
            CP_LENGTH:CP_LENGTH + NUM_SUBCARRIERS
        ]

        received_frequency = (
            np.fft.fft(received_no_cp)
            / np.sqrt(NUM_SUBCARRIERS)
        )

        # LMMSE channel estimation
        received_pilots = received_frequency[PILOT_INDICES]

        channel_estimate = phy.lmmse_estimation(
            received_pilots, noise_variance
        )

        # Equalization
        data_estimate = channel_estimate[DATA_INDICES]
        received_data = received_frequency[DATA_INDICES]

        safe_estimate = np.where(
            np.abs(data_estimate) < 1e-10,
            1e-10 + 0j,
            data_estimate
        )

        equalized = received_data / safe_estimate

        detected_bits = phy.demodulate(
            equalized, modulation
        )

        # Ignore padding bits in the final OFDM symbol
        detected_bits = detected_bits[:valid_bits]

        total_errors += np.count_nonzero(
            detected_bits != symbol_bits
        )

        bit_position += valid_bits

    packet_ber = total_errors / PACKET_BITS
    packet_error = int(total_errors > 0)

    return {
        "Packet BER": packet_ber,
        "Packet Error": packet_error,
        "Bit Errors": int(total_errors),
        "OFDM Symbols": number_of_symbols
    }


def wilson_interval(errors, total, z=1.96):
    """95% Wilson confidence interval for a packet error ratio."""

    if total == 0:
        return np.nan, np.nan

    p = errors / total
    denominator = 1 + z**2 / total

    centre = (
        p + z**2 / (2 * total)
    ) / denominator

    margin = (
        z * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        ) / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def run_experiment():

    records = []

    print("\nV3 PACKET-LEVEL PHY CALIBRATION")
    print("================================")
    print("Packet size:", PACKET_BYTES, "bytes")
    print("Seeds:", SEEDS)
    print("Packets per seed:", PACKETS_PER_SEED)
    print("Channel: 8-tap Rayleigh block fading")
    print("Estimator: LMMSE")
    print("Coding: Uncoded")
    print("Maximum retransmissions: Not applied in this experiment")

    for snr_db in SNR_VALUES:

        for modulation in MODULATIONS:

            print(
                f"\nTesting {modulation} at {snr_db} dB..."
            )

            for seed in SEEDS:

                phy.rng = np.random.default_rng(seed)

                for packet_number in range(PACKETS_PER_SEED):

                    result = transmit_one_packet(
                        snr_db, modulation
                    )

                    records.append({
                        "Seed": seed,
                        "Packet Number": packet_number + 1,
                        "Packet Bytes": PACKET_BYTES,
                        "SNR (dB)": snr_db,
                        "Modulation": modulation,
                        **result
                    })

    frames_df = pd.DataFrame(records)

    frames_path = (
        RESULTS_DIR / "v3_packet_phy_calibration_packets.csv"
    )

    frames_df.to_csv(frames_path, index=False)

    # Aggregate packet results
    summary_records = []

    for (snr_db, modulation), group in frames_df.groupby(
        ["SNR (dB)", "Modulation"]
    ):

        total_packets = len(group)
        packet_errors = int(group["Packet Error"].sum())

        per = packet_errors / total_packets

        lower, upper = wilson_interval(
            packet_errors, total_packets
        )

        summary_records.append({
            "SNR (dB)": snr_db,
            "Modulation": modulation,
            "Packets": total_packets,
            "Packet Errors": packet_errors,
            "Packet PER": per,
            "PER 95% CI Lower": lower,
            "PER 95% CI Upper": upper,
            "Average BER": group["Packet BER"].mean(),
            "Average OFDM Symbols": group["OFDM Symbols"].mean()
        })

    summary_df = pd.DataFrame(summary_records)

    summary_path = (
        RESULTS_DIR / "v3_packet_phy_calibration_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\nPACKET-LEVEL CALIBRATION SUMMARY")
    print("================================")

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nSaved packet records:", frames_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    run_experiment()
