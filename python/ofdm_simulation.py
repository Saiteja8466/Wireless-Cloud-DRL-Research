
import numpy as np
import matplotlib.pyplot as plt

# OFDM Configuration
N = 64
CP = 16
NUM_SYMBOLS = 1000

def qpsk_modulation(bits):
    bits = bits.reshape(-1, 2)
    real = 1 - 2 * bits[:, 0]
    imag = 1 - 2 * bits[:, 1]
    return (real + 1j * imag) / np.sqrt(2)

def qpsk_demodulation(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[:, 0] = (symbols.real < 0).astype(int)
    bits[:, 1] = (symbols.imag < 0).astype(int)
    return bits.flatten()

def ofdm_transmission(snr_db):
    # Generate random information bits
    bits = np.random.randint(
        0, 2, N * NUM_SYMBOLS * 2
    )

    # QPSK modulation
    modulated = qpsk_modulation(bits)
    symbols = modulated.reshape(NUM_SYMBOLS, N)

    # OFDM transmitter: IFFT
    time_signal = np.fft.ifft(
        symbols, axis=1
    ) * np.sqrt(N)

    # Add cyclic prefix
    transmitted = np.concatenate(
        [time_signal[:, -CP:], time_signal],
        axis=1
    )

    # AWGN channel
    ebn0 = 10 ** (snr_db / 10)
    noise_variance = 1 / (2 * ebn0)

    noise = np.sqrt(noise_variance / 2) * (
        np.random.randn(*transmitted.shape)
        + 1j * np.random.randn(*transmitted.shape)
    )

    received = transmitted + noise

    # OFDM receiver: Remove cyclic prefix
    received = received[:, CP:]

    # FFT
    received_symbols = np.fft.fft(
        received, axis=1
    ) / np.sqrt(N)

    # QPSK demodulation
    received_bits = qpsk_demodulation(
        received_symbols.flatten()
    )

    # BER calculation
    errors = np.sum(bits != received_bits)
    ber = errors / len(bits)

    return ber

# Run simulation
snr_values = [0, 5, 10, 15, 20]
ber_values = []

print("OFDM SIMULATION RESULTS")
print("-----------------------")

for snr in snr_values:
    ber = ofdm_transmission(snr)
    ber_values.append(ber)

    print(f"SNR: {snr} dB | BER: {ber:.6f}")

# Plot BER vs SNR
plt.figure(figsize=(8, 5))
plt.semilogy(snr_values, ber_values, 'o-')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("OFDM QPSK Performance over AWGN Channel")
plt.grid(True, which="both")
plt.savefig("results/ofdm_ber_awgn.png")
plt.show()
