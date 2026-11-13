
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ============================================================
# V3 PACKET-LEVEL SEED STATISTICAL ANALYSIS
# ============================================================

INPUT_FILE = Path(
    "results/v3/v3_packet_phy_calibration_packets_1000.csv"
)

OUTPUT_DIR = Path("results/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "Seed",
    "SNR (dB)",
    "Modulation",
    "Packet Error"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# PER for each independent seed and condition
seed_results = (
    df.groupby(["SNR (dB)", "Modulation", "Seed"])
    .agg(
        Packets=("Packet Error", "count"),
        Packet_Errors=("Packet Error", "sum"),
        PER=("Packet Error", "mean")
    )
    .reset_index()
)

# Aggregate the five seed-level PER values
summary_records = []

for (snr, modulation), group in seed_results.groupby(
    ["SNR (dB)", "Modulation"]
):

    values = group["PER"].to_numpy()
    n = len(values)

    mean_per = np.mean(values)
    std_per = np.std(values, ddof=1)

    # Student-t confidence interval across five seeds
    critical = stats.t.ppf(0.975, df=n - 1)
    margin = critical * std_per / np.sqrt(n)

    summary_records.append({
        "SNR (dB)": snr,
        "Modulation": modulation,
        "Seeds": n,
        "Mean PER": mean_per,
        "Seed PER SD": std_per,
        "95% CI Lower": max(0.0, mean_per - margin),
        "95% CI Upper": min(1.0, mean_per + margin)
    })

summary = pd.DataFrame(summary_records)

# Save seed-level and aggregate results
seed_path = OUTPUT_DIR / "v3_packet_seed_level_results.csv"
summary_path = OUTPUT_DIR / "v3_packet_seed_statistical_summary.csv"

seed_results.to_csv(seed_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nV3 SEED-LEVEL STATISTICAL ANALYSIS")
print("=================================")

print("\nPER for each seed:")
print(seed_results.to_string(index=False))

print("\nFive-seed statistical summary:")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSaved seed-level results:", seed_path)
print("Saved statistical summary:", summary_path)

print("\nANALYSIS COMPLETED")
