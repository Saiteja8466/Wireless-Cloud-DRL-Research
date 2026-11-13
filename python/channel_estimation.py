
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# OFDM CHANNEL ESTIMATION: LS vs LMMSE
# ==========================================

np.random.seed(42)

N = 64
CP = 16
NUM_FRAMES = 2000
NUM_TAPS = 8

SNR_DB = [0, 5, 10, 15, 20]

# Normalized multipath power profile
PDP = np.exp(-np.arange(NUM_TAPS) / 2.0)
PDP = PDP / np.sum(PDP)


def qpsk_mod(bits):
    bits = bits.reshape(-1, 2)
    return (
        (1 - 2 * bits[:, 0])
        + 1j * (1 - 2 * bits[:, 1])
    ) / np.sqrt(2)


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[:, 0] = (symbols.real < 0).astype(int)
    bits[:, 1] = (symbols.imag < 0).astype(int)
    return bits.flatten()


def generate_channel():
    h = (
        np.random.randn(NUM_TAPS)
        + 1j * np.random.randn(NUM_TAPS)
    ) / np.sqrt(2)

    return h * np.sqrt(PDP)


def channel_covariance():
    # Frequency-domain channel covariance
    delays = np.arange(NUM_TAPS)

    frequencies = np.arange(N)

    phase = np.exp(
        -2j * np.pi * np.outer(frequencies, delays) / N
    )

    return (phase * PDP) @ phase.conj().T


R_HH = channel_covariance()


def transmit_ofdm(symbols, h, noise_variance):
    time_signal = np.fft.ifft(symbols) * np.sqrt(N)

    tx_cp = np.concatenate(
        [time_signal[-CP:], time_signal]
    )

    received = np.convolve(tx_cp, h)

    received = received[:N + CP]

    noise = np.sqrt(noise_variance / 2) * (
        np.random.randn(N + CP)
        + 1j * np.random.randn(N + CP)
    )

    received += noise

    received = received[CP:]

    return np.fft.fft(received) / np.sqrt(N)


def simulate(snr_db):

    gamma = 10 ** (snr_db / 10)

    # Complex noise variance with CP overhead
    noise_variance = (1 + CP / N) / (2 * gamma)

    # Pilot symbols: known QPSK sequence
    pilot_bits = np.random.randint(0, 2, 2 * N)
    pilot = qpsk_mod(pilot_bits)

    ber = {"Perfect CSI": 0, "LS": 0, "LMMSE": 0}
    nmse = {"LS": 0.0, "LMMSE": 0.0}

    total_bits = 0
    channel_power = 0.0

    # LMMSE estimator matrix
    estimator = R_HH @ np.linalg.inv(
        R_HH + noise_variance * np.eye(N)
    )

    for _ in range(NUM_FRAMES):

        h = generate_channel()

        H = np.fft.fft(h, N)

        # Transmit pilot OFDM symbol
        Yp = transmit_ofdm(
            pilot, h, noise_variance
        )

        # LS channel estimate
        H_ls = Yp / pilot

        # LMMSE channel estimate
        H_lmmse = estimator @ H_ls

        # Generate data
        bits = np.random.randint(0, 2, 2 * N)
        data = qpsk_mod(bits)

        # Transmit data OFDM symbol
        Y = transmit_ofdm(
            data, h, noise_variance
        )

        # Equalization
        estimates = {
            "Perfect CSI": Y / H,
            "LS": Y / H_ls,
            "LMMSE": Y / H_lmmse
        }

        for method, equalized in estimates.items():

            received_bits = qpsk_demod(equalized)

            ber[method] += np.sum(
                bits != received_bits
            )

        nmse["LS"] += np.sum(
            np.abs(H_ls - H) ** 2
        )

        nmse["LMMSE"] += np.sum(
            np.abs(H_lmmse - H) ** 2
        )

        channel_power += np.sum(np.abs(H) ** 2)

        total_bits += len(bits)

    ber = {
        method: errors / total_bits
        for method, errors in ber.items()
    }

    denominator = NUM_FRAMES * channel_power / NUM_FRAMES

    nmse = {
        method: value / denominator
        for method, value in nmse.items()
    }

    return ber, nmse


# ==========================================
# RUN EXPERIMENTS
# ==========================================

results = {
    "Perfect CSI": [],
    "LS": [],
    "LMMSE": []
}

nmse_results = {
    "LS": [],
    "LMMSE": []
}

print("\nCHANNEL ESTIMATION RESULTS")
print("===========================")

for snr in SNR_DB:

    ber, nmse = simulate(snr)

    for method in results:
        results[method].append(ber[method])

    for method in nmse_results:
        nmse_results[method].append(nmse[method])

    print(f"\nSNR: {snr} dB")

    for method in results:
        print(f"{method} BER: {ber[method]:.6e}")

    for method in nmse_results:
        print(f"{method} NMSE: {nmse[method]:.6e}")


# ==========================================
# SAVE GRAPHS
# ==========================================

Path("results").mkdir(exist_ok=True)

plt.figure(figsize=(9, 6))

for method, values in results.items():
    plt.semilogy(SNR_DB, values, 'o-', label=method)

plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("OFDM Channel Estimation: BER Comparison")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("results/channel_estimation_ber.png", dpi=300)
plt.show()


plt.figure(figsize=(9, 6))

for method, values in nmse_results.items():
    plt.semilogy(SNR_DB, values, 's-', label=method)

plt.xlabel("SNR (dB)")
plt.ylabel("NMSE")
plt.title("LS vs LMMSE Channel Estimation Accuracy")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("results/channel_estimation_nmse.png", dpi=300)
plt.show()

print("\nChannel estimation experiment completed!")
