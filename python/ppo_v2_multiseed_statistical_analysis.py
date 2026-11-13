import os
import numpy as np
import pandas as pd
from scipy import stats

PPO_FILE = "results/ppo_v2_multiseed_raw_results.csv"
BASELINE_FILE = "results/ppo_v2_heldout_raw_results.csv"

RAW_OUT = "results/ppo_v2_multiseed_paired_raw.csv"
SEED_OUT = "results/ppo_v2_multiseed_seed_differences.csv"
SUMMARY_OUT = "results/ppo_v2_multiseed_statistical_summary.csv"

OUTPUTS = [RAW_OUT, SEED_OUT, SUMMARY_OUT]

# Protect existing files from accidental overwriting.
for path in OUTPUTS:
    if os.path.exists(path):
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Do not overwrite existing research results."
        )

ppo = pd.read_csv(PPO_FILE)
baseline = pd.read_csv(BASELINE_FILE)

metrics = {
    "Makespan_s": "Makespan_s",
    "Average_Response_s": "Average_Response_s",
    "Deadline_Miss_Rate_pct": "Deadline_Miss_Rate_pct",
    "Average_Energy_J": "Average_Energy_J"
}

# Validate input columns.
required_ppo = {"Training_Seed", "Seed", "Scheduler", *metrics.keys()}
required_baseline = {"Seed", "Scheduler", *metrics.values()}

if not required_ppo.issubset(ppo.columns):
    raise ValueError("PPO dataset is missing required columns.")

if not required_baseline.issubset(baseline.columns):
    raise ValueError("Baseline dataset is missing required columns.")

if ppo.duplicated(["Training_Seed", "Seed"]).any():
    raise ValueError("Duplicate PPO training/workload records found.")

if baseline.duplicated(["Scheduler", "Seed"]).any():
    raise ValueError("Duplicate baseline/workload records found.")

if set(ppo["Scheduler"].unique()) != {"PPO"}:
    raise ValueError("Unexpected scheduler in PPO dataset.")

training_seeds = sorted(ppo["Training_Seed"].unique())
test_seeds = sorted(ppo["Seed"].unique())

if len(training_seeds) != 5:
    raise ValueError("Expected exactly five independent training seeds.")

if len(test_seeds) != 20:
    raise ValueError("Expected exactly twenty held-out workload seeds.")

counts = ppo.groupby("Training_Seed")["Seed"].count()

if not (counts == 20).all():
    raise ValueError("Each PPO training seed must have twenty evaluations.")

baselines = sorted(
    baseline.loc[baseline["Scheduler"] != "PPO", "Scheduler"].unique()
)

if len(baselines) != 5:
    raise ValueError("Expected exactly five baseline schedulers.")

baseline = baseline[baseline["Scheduler"].isin(baselines)].copy()

if set(baseline["Seed"].unique()) != set(test_seeds):
    raise ValueError("PPO and baseline workload seeds do not match.")

if not np.isfinite(ppo[list(metrics.keys())].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite PPO values found.")

if not np.isfinite(
    baseline[list(metrics.values())].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite baseline values found.")

print("Input validation passed.")
print("Training seeds:", training_seeds)
print("Held-out workloads:", len(test_seeds))
print("Baseline schedulers:", baselines)

# Pair PPO and each baseline on identical workload seeds.
records = []

for train_seed in training_seeds:
    ppo_model = ppo[ppo["Training_Seed"] == train_seed]

    for scheduler in baselines:
        base_model = baseline[baseline["Scheduler"] == scheduler]

        merged = ppo_model.merge(
            base_model,
            on="Seed",
            suffixes=("_PPO", "_Baseline"),
            validate="one_to_one"
        )

        if len(merged) != 20:
            raise ValueError(
                f"Expected 20 paired workloads for "
                f"training seed {train_seed} vs {scheduler}; "
                f"found {len(merged)}."
            )

        for metric, baseline_metric in metrics.items():
            for _, row in merged.iterrows():
                ppo_value = float(row[f"{metric}_PPO"])
                baseline_value = float(row[f"{baseline_metric}_Baseline"])

                records.append({
                    "Training_Seed": train_seed,
                    "Test_Seed": int(row["Seed"]),
                    "Baseline": scheduler,
                    "Metric": metric,
                    "PPO_Value": ppo_value,
                    "Baseline_Value": baseline_value,
                    "PPO_Minus_Baseline": ppo_value - baseline_value
                })

raw = pd.DataFrame(records)

# Workload-level paired differences averaged within each trained model.
seed_level = (
    raw.groupby(["Training_Seed", "Baseline", "Metric"])
    ["PPO_Minus_Baseline"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={
        "mean": "Mean_Paired_Difference",
        "std": "Workload_Difference_Std"
    })
)

# Statistical tests use five independent training runs as observations.
results = []

for (scheduler, metric), group in seed_level.groupby(
    ["Baseline", "Metric"]
):
    differences = group["Mean_Paired_Difference"].to_numpy(dtype=float)

    n = len(differences)
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    se = sd_diff / np.sqrt(n)

    t_stat, t_p = stats.ttest_1samp(differences, popmean=0)

    if np.allclose(differences, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            method="exact"
        )
        w_stat = float(wilcoxon_result.statistic)
        w_p = float(wilcoxon_result.pvalue)

    critical_t = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - critical_t * se
    ci_high = mean_diff + critical_t * se

    cohens_dz = mean_diff / sd_diff if sd_diff > 0 else np.nan

    results.append({
        "Baseline": scheduler,
        "Metric": metric,
        "Independent_Training_Runs": n,
        "Mean_Paired_Difference": mean_diff,
        "SD_Across_Training_Runs": sd_diff,
        "CI95_Lower": ci_low,
        "CI95_Upper": ci_high,
        "Cohens_dz": cohens_dz,
        "Paired_t_statistic": float(t_stat),
        "Paired_t_p": float(t_p),
        "Wilcoxon_statistic": w_stat,
        "Wilcoxon_p": float(w_p)
    })

summary = pd.DataFrame(results)

# Holm-Bonferroni correction across the 20 comparisons,
# separately for the paired t-test and Wilcoxon test families.
def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)

    return adjusted

summary["Paired_t_Holm_p"] = holm_adjust(summary["Paired_t_p"])
summary["Wilcoxon_Holm_p"] = holm_adjust(summary["Wilcoxon_p"])

summary["Direction"] = np.where(
    summary["Mean_Paired_Difference"] < 0,
    "PPO lower",
    np.where(
        summary["Mean_Paired_Difference"] > 0,
        "PPO higher",
        "Equal"
    )
)

# Save three new files only.
raw.to_csv(RAW_OUT, index=False)
seed_level.to_csv(SEED_OUT, index=False)
summary.to_csv(SUMMARY_OUT, index=False)

print("\n" + "=" * 75)
print("MULTI-SEED PAIRED STATISTICAL ANALYSIS COMPLETED")
print("=" * 75)

display_columns = [
    "Baseline",
    "Metric",
    "Mean_Paired_Difference",
    "CI95_Lower",
    "CI95_Upper",
    "Paired_t_Holm_p",
    "Wilcoxon_Holm_p",
    "Direction"
]

print(summary[display_columns].to_string(index=False))

print("\nFILES SAVED:")
for path in OUTPUTS:
    print(path)

print("\nStatistical unit: five independent PPO training runs.")
print("Workload differences were paired and averaged within each run.")

