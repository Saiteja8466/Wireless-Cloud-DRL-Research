import pandas as pd
from pathlib import Path

# ============================================================
# V3 PACKETIZATION ANALYSIS
# ============================================================

RESULTS_DIR = Path("results/v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_DATA_SUBCARRIERS = 48
MAX_ARQ_ATTEMPTS = 4

MODULATION_BITS = {
    "BPSK": 1,
    "QPSK": 2,
    "16-QAM": 4,
    "64-QAM": 6
}

TASK_SIZES_MB = [0.1, 0.5, 1.0, 1.5, 2.0]

# Empirical PER values from the five-seed PHY validation.
CALIBRATED_LINKS = [
    (16, "BPSK", 0.25040),
    (20, "QPSK", 0.31560),
    (26, "16-QAM", 0.42240),
    (30, "16-QAM", 0.23920)
]

records = []

for task_size_mb in TASK_SIZES_MB:

    payload_bits = int(task_size_mb * 8_000_000)

    for snr_db, modulation, per in CALIBRATED_LINKS:

        bits_per_symbol = MODULATION_BITS[modulation]

        bits_per_ofdm_symbol = (
            NUM_DATA_SUBCARRIERS * bits_per_symbol
        )

        required_symbols = (
            payload_bits + bits_per_ofdm_symbol - 1
        ) // bits_per_ofdm_symbol

        # Expected number of attempts per frame under
        # independent block-fading ARQ with a four-attempt cap.
        expected_attempts_per_frame = sum(
            per ** attempt
            for attempt in range(MAX_ARQ_ATTEMPTS)
        )

        frame_success_probability = (
            1 - per ** MAX_ARQ_ATTEMPTS
        )

        # Independent-frame approximation:
        # probability that every frame in the task succeeds.
        task_success_probability = (
            frame_success_probability ** required_symbols
        )

        records.append({
            "Task Size (MB)": task_size_mb,
            "SNR (dB)": snr_db,
            "Modulation": modulation,
            "Empirical Frame PER": per,
            "Payload Bits": payload_bits,
            "Bits per OFDM Symbol": bits_per_ofdm_symbol,
            "Required OFDM Symbols": required_symbols,
            "Expected Attempts per Frame": expected_attempts_per_frame,
            "Frame Success Probability": frame_success_probability,
            "Estimated Whole Task Success Probability": task_success_probability
        })

df = pd.DataFrame(records)

output = RESULTS_DIR / "v3_packetization_analysis.csv"
df.to_csv(output, index=False)

print("\nV3 PACKETIZATION ANALYSIS")
print("=========================")

print(
    df[[
        "Task Size (MB)",
        "SNR (dB)",
        "Modulation",
        "Required OFDM Symbols",
        "Expected Attempts per Frame",
        "Estimated Whole Task Success Probability"
    ]].to_string(index=False)
)

print(f"\nSaved: {output}")
