
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SIMULATION CONFIGURATION
# ==========================================

rng = np.random.default_rng(42)

SNR_DB = [0, 5, 10, 15, 20]

NUM_PACKETS = 500
BITS_PER_PACKET = 1024
MAX_RETRANSMISSIONS = 3

# QPSK: 2 bits per symbol
BITS_PER_SYMBOL = 2
SYMBOLS_PER_PACKET = BITS_PER_PACKET // BITS_PER_SYMBOL

# ==========================================
# 2. QPSK MODULATION
# ==========================================

def qpsk_modulate(bits):

    bit_pairs = bits.reshape(-1, 2)

    symbols = (
        (1 - 2 * bit_pairs[:, 0])
        + 1j * (1 - 2 * bit_pairs[:, 1])
    ) / np.sqrt(2)

    return symbols


def qpsk_demodulate(symbols):

    detected = np.empty(
        (len(symbols), 2),
        dtype=int
    )

    detected[:, 0] = symbols.real < 0
    detected[:, 1] = symbols.imag < 0

    return detected.ravel()


# ==========================================
# 3. ONE PACKET TRANSMISSION
# ==========================================

def transmit_packet(snr_db, estimator):

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = 1 / snr_linear

    # Generate random packet
    bits = rng.integers(
        0, 2, BITS_PER_PACKET
    )

    tx_symbols = qpsk_modulate(bits)

    # Rayleigh block fading channel
    h = (
        rng.normal()
        + 1j * rng.normal()
    ) / np.sqrt(2)

    # Pilot transmission
    pilot_noise = np.sqrt(
        noise_variance / 2
    ) * (
        rng.normal()
        + 1j * rng.normal()
    )

    pilot_received = h + pilot_noise

    # Channel estimation
    if estimator == "Perfect CSI":

        h_estimated = h

    elif estimator == "LS":

        h_estimated = pilot_received

    elif estimator == "LMMSE":

        h_estimated = (
            pilot_received / (1 + noise_variance)
        )

    else:
        raise ValueError("Unknown channel estimator")

    # Data transmission
    data_noise = np.sqrt(
        noise_variance / 2
    ) * (
        rng.normal(size=len(tx_symbols))
        + 1j * rng.normal(size=len(tx_symbols))
    )

    received = h * tx_symbols + data_noise

    # Equalization
    if np.abs(h_estimated) < 1e-12:

        detected_bits = np.zeros(
            BITS_PER_PACKET,
            dtype=int
        )

    else:

        equalized = received / h_estimated

        detected_bits = qpsk_demodulate(
            equalized
        )

    # Packet error detection
    errors = np.sum(bits != detected_bits)

    packet_error = errors > 0

    return packet_error, errors


# ==========================================
# 4. ARQ RETRANSMISSION SIMULATION
# ==========================================

def simulate_arq(snr_db, estimator):

    successful_packets = 0
    total_attempts = 0
    first_attempt_errors = 0

    for packet_index in range(NUM_PACKETS):

        packet_delivered = False

        for attempt in range(
            MAX_RETRANSMISSIONS + 1
        ):

            packet_error, errors = transmit_packet(
                snr_db,
                estimator
            )

            total_attempts += 1

            if attempt == 0 and packet_error:
                first_attempt_errors += 1

            if not packet_error:

                successful_packets += 1
                packet_delivered = True
                break

    delivery_ratio = (
        successful_packets / NUM_PACKETS
    )

    first_attempt_per = (
        first_attempt_errors / NUM_PACKETS
    )

    average_attempts = (
        total_attempts / NUM_PACKETS
    )

    # Each attempt includes one pilot symbol
    symbols_per_attempt = (
        SYMBOLS_PER_PACKET + 1
    )

    delivered_bits = (
        successful_packets * BITS_PER_PACKET
    )

    total_channel_uses = (
        total_attempts * symbols_per_attempt
    )

    goodput = (
        delivered_bits / total_channel_uses
        if total_channel_uses > 0
        else 0
    )

    return {
        "SNR (dB)": snr_db,
        "Estimator": estimator,
        "First Attempt PER": first_attempt_per,
        "Delivery Ratio (%)": delivery_ratio * 100,
        "Average Attempts": average_attempts,
        "Goodput (bits/channel use)": goodput
    }


# ==========================================
# 5. RUN ALL EXPERIMENTS
# ==========================================

estimators = [
    "Perfect CSI",
    "LS",
    "LMMSE"
]

results = []

for estimator in estimators:

    print(f"\n{estimator} Results")
    print("=" * 45)

    for snr_db in SNR_DB:

        metrics = simulate_arq(
            snr_db,
            estimator
        )

        results.append(metrics)

        print(
            f"SNR: {snr_db:2d} dB | "
            f"First Attempt PER: "
            f"{metrics['First Attempt PER']:.4f} | "
            f"Delivery Ratio: "
            f"{metrics['Delivery Ratio (%)']:.2f}% | "
            f"Average Attempts: "
            f"{metrics['Average Attempts']:.3f} | "
            f"Goodput: "
            f"{metrics['Goodput (bits/channel use)']:.4f}"
        )


# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/channel_reliability_arq.csv",
    index=False
)


# ==========================================
# 7. PLOT PACKET ERROR RATE
# ==========================================

plt.figure(figsize=(9, 5))

for estimator in estimators:

    subset = results_df[
        results_df["Estimator"] == estimator
    ]

    plt.semilogy(
        subset["SNR (dB)"],
        subset["First Attempt PER"],
        marker="o",
        label=estimator
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Packet Error Rate")
plt.title("Channel Estimation: Packet Error Performance")

plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/channel_reliability_per.png",
    dpi=300
)

plt.show()


# ==========================================
# 8. PLOT DELIVERY RATIO
# ==========================================

plt.figure(figsize=(9, 5))

for estimator in estimators:

    subset = results_df[
        results_df["Estimator"] == estimator
    ]

    plt.plot(
        subset["SNR (dB)"],
        subset["Delivery Ratio (%)"],
        marker="o",
        label=estimator
    )

plt.xlabel("SNR (dB)")
plt.ylabel("Successful Packet Delivery (%)")
plt.title("ARQ Packet Delivery Performance")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/channel_reliability_delivery.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. COMPLETION
# ==========================================

print("\nChannel reliability and ARQ experiment completed!")
