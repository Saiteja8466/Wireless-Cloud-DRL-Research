
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ============================================================
# V3 PACKET-LEVEL ARQ VALIDATION
# ============================================================

INPUT_FILE = Path(
    "results/v3/v3_packet_seed_level_results.csv"
)

OUTPUT_DIR = Path("results/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ATTEMPTS = 4
ARQ_PACKETS_PER_CONDITION = 10000

EXPECTED_SEEDS = [7, 42, 99, 123, 2026]

MODULATION_ORDER = [
    "BPSK", "QPSK", "16-QAM", "64-QAM"
]

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "SNR (dB)", "Modulation", "Seed", "PER"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

records = []

print("\nV3 PACKET-LEVEL ARQ VALIDATION")
print("=============================")
print("Maximum attempts:", MAX_ATTEMPTS)
print("Packets per seed and condition:", ARQ_PACKETS_PER_CONDITION)
print("Retransmission channel assumption: Independent block fading")
print("Payload: Same packet retained across retries")
print("ARQ model: Calibrated packet PER")

# ============================================================
# RUN ARQ MONTE CARLO EXPERIMENT
# ============================================================

for _, row in df.iterrows():

    snr = int(row["SNR (dB)"])
    modulation = row["Modulation"]
    seed = int(row["Seed"])
    per = float(row["PER"])

    if not 0 <= per <= 1:
        raise ValueError(f"Invalid PER value: {per}")

    # Independent deterministic random stream for each condition
    modulation_index = MODULATION_ORDER.index(modulation)

    rng = np.random.default_rng(
        np.random.SeedSequence([
            seed, snr, modulation_index, 4401
        ])
    )

    # True means that an attempt successfully delivers the packet.
    success_matrix = (
        rng.random(
            (ARQ_PACKETS_PER_CONDITION, MAX_ATTEMPTS)
        ) >= per
    )

    delivered = success_matrix.any(axis=1)

    # First successful attempt determines the attempt count.
    first_success = np.argmax(
        success_matrix, axis=1
    )

    attempts = np.full(
        ARQ_PACKETS_PER_CONDITION,
        MAX_ATTEMPTS,
        dtype=int
    )

    attempts[delivered] = first_success[delivered] + 1

    delivered_count = int(delivered.sum())
    dropped_count = int((~delivered).sum())

    average_attempts = float(attempts.mean())

    records.append({
        "Seed": seed,
        "SNR (dB)": snr,
        "Modulation": modulation,
        "Input Packet PER": per,
        "ARQ Packets": ARQ_PACKETS_PER_CONDITION,
        "Delivered Packets": delivered_count,
        "Dropped Packets": dropped_count,
        "Delivery Ratio": delivered.mean(),
        "Drop Ratio": (~delivered).mean(),
        "Average Attempts": average_attempts,
        "Average Retransmissions": attempts.mean() - 1,
        "Theoretical Delivery Probability":
            1 - per ** MAX_ATTEMPTS,
        "Theoretical Expected Attempts":
            sum(per ** k for k in range(MAX_ATTEMPTS))
    })

    print(
        f"SNR {snr:2d} dB | {modulation:6s} | "
        f"Seed {seed:4d} | "
        f"Delivery: {delivered.mean():.4f} | "
        f"Attempts: {average_attempts:.4f}"
    )

seed_results = pd.DataFrame(records)

# ============================================================
# SAVE SEED-LEVEL ARQ RESULTS
# ============================================================

seed_path = (
    OUTPUT_DIR / "v3_packet_arq_seed_results.csv"
)

seed_results.to_csv(seed_path, index=False)

# ============================================================
# FIVE-SEED STATISTICAL SUMMARY
# ============================================================

summary_records = []

metric_columns = [
    "Delivery Ratio",
    "Drop Ratio",
    "Average Attempts",
    "Average Retransmissions"
]

for (snr, modulation), group in seed_results.groupby(
    ["SNR (dB)", "Modulation"]
):

    result = {
        "SNR (dB)": snr,
        "Modulation": modulation,
        "Seeds": len(group)
    }

    for metric in metric_columns:

        values = group[metric].to_numpy()

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)

        n = len(values)

        if n > 1:
            critical = stats.t.ppf(
                0.975, df=n - 1
            )

            margin = (
                critical * std_value / np.sqrt(n)
            )
        else:
            margin = np.nan

        result[f"Mean {metric}"] = mean_value
        result[f"SD {metric}"] = std_value
        result[f"{metric} CI Lower"] = (
            max(0.0, mean_value - margin)
            if metric in ["Delivery Ratio", "Drop Ratio"]
            else mean_value - margin
        )
        result[f"{metric} CI Upper"] = (
            min(1.0, mean_value + margin)
            if metric in ["Delivery Ratio", "Drop Ratio"]
            else mean_value + margin
        )

    summary_records.append(result)

summary = pd.DataFrame(summary_records)

summary_path = (
    OUTPUT_DIR / "v3_packet_arq_statistical_summary.csv"
)

summary.to_csv(summary_path, index=False)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nARQ FIVE-SEED STATISTICAL SUMMARY")
print("=================================")

display_columns = [
    "SNR (dB)",
    "Modulation",
    "Seeds",
    "Mean Delivery Ratio",
    "Mean Drop Ratio",
    "Mean Average Attempts",
    "Mean Average Retransmissions"
]

print(
    summary[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved seed results:", seed_path)
print("Saved statistical summary:", summary_path)

print("\nV3 ARQ VALIDATION COMPLETED")
