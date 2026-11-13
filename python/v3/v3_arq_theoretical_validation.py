
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# V3 ARQ THEORETICAL VS MONTE CARLO VALIDATION
# ============================================================

INPUT_FILE = Path(
    "results/v3/v3_packet_arq_seed_results_final.csv"
)

OUTPUT_DIR = Path("results/v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ATTEMPTS = 4

df = pd.read_csv(INPUT_FILE)

required = {
    "Seed",
    "SNR (dB)",
    "Modulation",
    "Input Packet PER",
    "Delivery Ratio",
    "Average Attempts"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Theoretical probability of delivery within four attempts
df["Theoretical Delivery"] = (
    1 - df["Input Packet PER"] ** MAX_ATTEMPTS
)

# Theoretical expected number of attempts, including
# packets that exhaust all four attempts.
df["Theoretical Attempts"] = sum(
    df["Input Packet PER"] ** k
    for k in range(MAX_ATTEMPTS)
)

df["Delivery Error"] = (
    df["Delivery Ratio"] - df["Theoretical Delivery"]
)

df["Attempts Error"] = (
    df["Average Attempts"] - df["Theoretical Attempts"]
)

# Aggregate five seeds
summary = (
    df.groupby(["SNR (dB)", "Modulation"])
    .agg(
        Seeds=("Seed", "count"),
        Mean_Input_PER=("Input Packet PER", "mean"),
        Mean_Simulated_Delivery=("Delivery Ratio", "mean"),
        Mean_Theoretical_Delivery=("Theoretical Delivery", "mean"),
        Mean_Delivery_Error=("Delivery Error", "mean"),
        Mean_Simulated_Attempts=("Average Attempts", "mean"),
        Mean_Theoretical_Attempts=("Theoretical Attempts", "mean"),
        Mean_Attempts_Error=("Attempts Error", "mean")
    )
    .reset_index()
)

# Absolute differences
summary["Absolute Delivery Difference"] = (
    summary["Mean_Delivery_Error"].abs()
)

summary["Absolute Attempts Difference"] = (
    summary["Mean_Attempts_Error"].abs()
)

output_file = (
    OUTPUT_DIR / "v3_arq_theoretical_validation.csv"
)

summary.to_csv(output_file, index=False)

print("\nV3 ARQ THEORETICAL VALIDATION")
print("============================")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nMaximum absolute delivery difference:",
      f"{summary['Absolute Delivery Difference'].max():.6f}")

print("Maximum absolute attempts difference:",
      f"{summary['Absolute Attempts Difference'].max():.6f}")

print("\nSaved:", output_file)
print("\nTHEORETICAL VALIDATION COMPLETED")
