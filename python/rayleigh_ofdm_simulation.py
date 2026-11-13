
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# OFDM RAYLEIGH FADING SIMULATION
# ==========================================

np.random.seed(42)

N = 64
CP = 16
NUM_SYMBOLS = 3000

SNR_DB = [0, 5, 10, 15, 20]

# QPSK modulation
def qpsk_modulation(bits):
    bits = bits.reshape(-1, 2)

    real = 1 - 2 * bits[:, 0]
    imag = 1 - 2 * bits[:, 1]

    return (real + 1j * imag) / np.sqrt(2)


# QPSK demodulation
def qpsk_demodulation(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)

    bits[:, 0] = (symbols.real < 0).astype(int)
    bits[:, 1] = (symbols.imag < 0).astype(int)

    return bits.flatten()


# Generate normalized multipath Rayleigh channel
def generate_rayleigh_channel():

    taps = 8

    power_profile = np.exp(
        -np.arange(taps) / 2.0
    )

    power_profile /= np.sum(power_profile)

    h = (
        np.random.randn(taps)
        + 1j * np.random.randn(taps)
    ) / np.sqrt(2)

    h *= np.sqrt(power_profile)

    return h


# OFDM simulation
def simulate_ofdm(snr_db, channel_type):

    bits = np.random.randint(
        0, 2, NUM_SYMBOLS * N * 2
    )

    modulated = qpsk_modulation(bits)

    symbols = modulated.reshape(NUM_SYMBOLS, N)

    # OFDM transmitter: IFFT
    tx = np.fft.ifft(
        symbols, axis=1
    ) * np.sqrt(N)

    # Add cyclic prefix
    tx_cp = np.concatenate(
        [tx[:, -CP:], tx],
        axis=1
    )

    # Eb/N0 noise variance, including CP overhead
    gamma = 10 ** (snr_db / 10)

    noise_variance = (1 + CP / N) / (2 * gamma)

    errors = 0

    for i in range(NUM_SYMBOLS):

        signal = tx_cp[i]

        if channel_type == "rayleigh":

            h = generate_rayleigh_channel()

            # Linear convolution
            received = np.convolve(signal, h)

            # Keep required OFDM symbol samples
            received = received[:N + CP]

            H = np.fft.fft(h, N)

        else:

            received = signal.copy()

            H = np.ones(N, dtype=complex)

        # Add complex AWGN
        noise = np.sqrt(noise_variance / 2) * (
            np.random.randn(N + CP)
            + 1j * np.random.randn(N + CP)
        )

        received += noise

        # Remove cyclic prefix
        received = received[CP:]

        # FFT receiver
        Y = np.fft.fft(received) / np.sqrt(N)

        # Perfect CSI frequency-domain equalization
        equalized = Y / H

        # Demodulation
        rx_bits = qpsk_demodulation(equalized)

        tx_bits = bits[i * N * 2:(i + 1) * N * 2]

        errors += np.sum(tx_bits != rx_bits)

    ber = errors / len(bits)

    return ber


# Run experiments
awgn_results = []
rayleigh_results = []

print("\nOFDM CHANNEL COMPARISON")
print("=======================")

for snr in SNR_DB:

    ber_awgn = simulate_ofdm(snr, "awgn")
    ber_rayleigh = simulate_ofdm(snr, "rayleigh")

    awgn_results.append(ber_awgn)
    rayleigh_results.append(ber_rayleigh)

    print(
        f"SNR: {snr:2d} dB | "
        f"AWGN BER: {ber_awgn:.6e} | "
        f"Rayleigh BER: {ber_rayleigh:.6e}"
    )


# Plot results
plt.figure(figsize=(9, 6))

plt.semilogy(
    SNR_DB, awgn_results,
    'o-', label="AWGN"
)

plt.semilogy(
    SNR_DB, rayleigh_results,
    's-', label="Rayleigh Fading"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("OFDM QPSK: AWGN vs Rayleigh Fading")

plt.grid(True, which="both")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/rayleigh_ofdm_comparison.png",
    dpi=300
)

plt.show()
