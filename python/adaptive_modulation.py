
import numpy as np
import matplotlib.pyplot as plt

# Reproducible simulation
np.random.seed(42)

# Simulation parameters
NUM_BITS = 200_000
SNR_DB = np.arange(0, 31, 2)


# Adaptive modulation selection
def select_modulation(snr_db):
    if snr_db < 8:
        return "BPSK", 1
    elif snr_db < 14:
        return "QPSK", 2
    elif snr_db < 21:
        return "16-QAM", 4
    else:
        return "64-QAM", 6


# Modulation function
def qam_modulate(bits, modulation):

    if modulation == "BPSK":
        return (1 - 2 * bits).astype(complex)

    if modulation == "QPSK":
        b = bits.reshape(-1, 2)
        symbols = (1 - 2*b[:, 0]) + 1j*(1 - 2*b[:, 1])
        return symbols / np.sqrt(2)

    M = 16 if modulation == "16-QAM" else 64
    k = int(np.log2(M))
    side = int(np.sqrt(M))

    b = bits.reshape(-1, k)
    weights = 2 ** np.arange(k - 1, -1, -1)
    indices = b @ weights

    gray = np.arange(side) ^ (np.arange(side) >> 1)
    levels = 2 * np.arange(side) - (side - 1)

    i_gray = indices >> (k // 2)
    q_gray = indices & (side - 1)

    i_level = levels[gray[i_gray]]
    q_level = levels[gray[q_gray]]

    scale = np.sqrt(2 * (M - 1) / 3)

    return (i_level + 1j * q_level) / scale


# Demodulation function
def qam_demodulate(symbols, modulation):

    if modulation == "BPSK":
        return (symbols.real < 0).astype(int)

    if modulation == "QPSK":
        out = np.empty((len(symbols), 2), dtype=int)
        out[:, 0] = symbols.real < 0
        out[:, 1] = symbols.imag < 0
        return out.ravel()

    M = 16 if modulation == "16-QAM" else 64
    k = int(np.log2(M))
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

    return ((indices[:, None] & weights) > 0).astype(int).ravel()


# Main adaptive modulation experiment
ber_results = []
spectral_efficiency = []
selected_modes = []

for snr_db in SNR_DB:

    modulation, bits_per_symbol = select_modulation(snr_db)

    num_bits = (
        NUM_BITS // bits_per_symbol
    ) * bits_per_symbol

    bits = np.random.randint(0, 2, num_bits)

    # Transmitter
    tx_symbols = qam_modulate(bits, modulation)

    # AWGN channel
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / snr_linear

    noise = np.sqrt(noise_variance / 2) * (
        np.random.randn(len(tx_symbols))
        + 1j * np.random.randn(len(tx_symbols))
    )

    rx_symbols = tx_symbols + noise

    # Receiver
    detected_bits = qam_demodulate(
        rx_symbols, modulation
    )

    # BER calculation
    ber = np.mean(bits != detected_bits)

    ber_results.append(ber)
    spectral_efficiency.append(bits_per_symbol)
    selected_modes.append(modulation)

    print(
        f"SNR: {snr_db:2d} dB | "
        f"Modulation: {modulation:6s} | "
        f"Spectral Efficiency: {bits_per_symbol} bits/symbol | "
        f"BER: {ber:.6e}"
    )


# BER performance graph
plt.figure(figsize=(9, 5))

plt.semilogy(
    SNR_DB,
    ber_results,
    marker="o"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Adaptive Modulation: BER Performance")

plt.grid(True, which="both")
plt.tight_layout()

plt.savefig(
    "results/adaptive_modulation_ber.png",
    dpi=300
)

plt.show()


# Modulation selection graph
plt.figure(figsize=(9, 5))

plt.step(
    SNR_DB,
    spectral_efficiency,
    where="mid",
    marker="o"
)

plt.xlabel("SNR (dB)")
plt.ylabel("Bits per Symbol")
plt.title("Adaptive Modulation Selection")

plt.yticks(
    [1, 2, 4, 6],
    ["BPSK", "QPSK", "16-QAM", "64-QAM"]
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/adaptive_modulation_selection.png",
    dpi=300
)

plt.show()

print("\nAdaptive modulation experiment completed!")
