import os
import numpy as np
import pandas as pd
from scipy import stats

V2_FILE = "results/ppo_v2_heldout_raw_results.csv"
V1_FILE = "results/ppo_paired_statistical_raw.csv"

OUT_RAW = "results/ppo_v2_paired_statistical_raw.csv"
OUT_SUMMARY = "results/ppo_v2_paired_statistical_summary.csv"
OUT_VERSION = "results/ppo_v1_vs_v2_paired_comparison.csv"

METRIC_MAP = {
    "Makespan_s": "Makespan_s",
    "Average_Response_s": "Average_Response_s",
    "Deadline_Miss_Rate_pct": "Deadline_Miss_Rate_pct",
    "Average_Energy_J": "Average_Energy_J"

}

BASELINES = [
    "FCFS",
    "Min-Min Offline",
    "Online Earliest Finish",
    "Random",
    "Round Robin"
]

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, index in enumerate(order):
        value = (n - rank) * p_values[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def paired_statistics(a, b, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    difference = a - b
    n = len(difference)

    mean_diff = np.mean(difference)
    sd_diff = np.std(difference, ddof=1)

    if sd_diff > 0:
        effect_size = mean_diff / sd_diff
    else:
        effect_size = np.nan

    t_result = stats.ttest_rel(a, b)

    if np.allclose(difference, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = stats.wilcoxon(
            difference,
            alternative="two-sided",
            method="auto"
        ).pvalue

    bootstrap_means = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(difference, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    ci_low, ci_high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    return {
        "N": n,
        "Mean_Difference": mean_diff,
        "SD_Difference": sd_diff,
        "95CI_Lower": ci_low,
        "95CI_Upper": ci_high,
        "Paired_t_p": t_result.pvalue,
        "Wilcoxon_p": wilcoxon_p,
        "Cohens_dz": effect_size
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 65)
    print("PPO V2 PAIRED STATISTICAL VALIDATION")
    print("=" * 65)

    v2 = pd.read_csv(V2_FILE)
    v1 = pd.read_csv(V1_FILE)

    required_v2 = {
        "Seed", "Scheduler", "Makespan_s",
        "Average_Response_s", "Deadline_Miss_Rate_pct",
        "Average_Energy_J"
    }

    required_v1 = {
        "Seed", "Metric", "PPO_Value"
    }

    if not required_v2.issubset(v2.columns):
        raise ValueError(
            f"Missing V2 columns: {required_v2 - set(v2.columns)}"
        )

    if not required_v1.issubset(v1.columns):
        raise ValueError(
            f"Missing V1 columns: {required_v1 - set(v1.columns)}"
        )

    if v2.duplicated(["Seed", "Scheduler"]).any():
        raise ValueError("Duplicate Seed/Scheduler rows in V2 data.")

    print("V2 rows:", len(v2))
    print("V1 statistical rows:", len(v1))

    v2_seeds = set(v2["Seed"].unique())
    v1_seeds = set(v1["Seed"].unique())

    common_seeds = sorted(v2_seeds.intersection(v1_seeds))

    if len(common_seeds) != 20:
        raise ValueError(
            f"Expected 20 common seeds, found {len(common_seeds)}."
        )

    print("Common held-out seeds:", common_seeds)

    v2 = v2[v2["Seed"].isin(common_seeds)].copy()

    # Reconstruct V1 PPO metrics from the earlier paired results.
    v1_ppo = {}

    for v2_metric, v1_metric in METRIC_MAP.items():
        subset = v1[v1["Metric"] == v1_metric]

        if subset.empty:
            raise ValueError(
                f"V1 metric not found: {v1_metric}. "
                f"Available metrics: {sorted(v1['Metric'].unique())}"
            )

        counts = subset.groupby("Seed")["PPO_Value"].nunique()

        if (counts > 1).any():
            raise ValueError(
                f"Inconsistent V1 PPO values for {v1_metric}."
            )

        values = subset.groupby("Seed")["PPO_Value"].first()

        if not set(common_seeds).issubset(set(values.index)):
            raise ValueError(
                f"Missing V1 seed values for {v1_metric}."
            )

        v1_ppo[v2_metric] = values

    print("V1 PPO values reconstructed and consistency-checked.")

    # V2 against all five baseline schedulers.
    raw_records = []

    for metric in METRIC_MAP:
        for baseline in BASELINES:
            ppo_rows = (
                v2[v2["Scheduler"] == "PPO"]
                .set_index("Seed")[metric]
                .reindex(common_seeds)
            )

            baseline_rows = (
                v2[v2["Scheduler"] == baseline]
                .set_index("Seed")[metric]
                .reindex(common_seeds)
            )

            if ppo_rows.isna().any() or baseline_rows.isna().any():
                raise ValueError(
                    f"Missing values for {metric}, {baseline}."
                )

            for seed in common_seeds:
                raw_records.append({
                    "Seed": seed,
                    "Metric": metric,
                    "Baseline": baseline,
                    "PPO_V2_Value": ppo_rows.loc[seed],
                    "Baseline_Value": baseline_rows.loc[seed],
                    "PPO_V2_Minus_Baseline":
                        ppo_rows.loc[seed] - baseline_rows.loc[seed]
                })

    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(OUT_RAW, index=False)

    # Statistical summaries and Holm correction across five baselines
    # separately for each metric and each statistical test.
    summary_records = []

    for metric in METRIC_MAP:
        metric_results = []

        for baseline in BASELINES:
            group = raw_df[
                (raw_df["Metric"] == metric) &
                (raw_df["Baseline"] == baseline)
            ]

            result = paired_statistics(
                group["PPO_V2_Value"],
                group["Baseline_Value"],
                rng
            )

            result.update({
                "Metric": metric,
                "Comparison": f"PPO V2 vs {baseline}"
            })

            metric_results.append(result)

        t_adjusted = holm_adjust(
            [x["Paired_t_p"] for x in metric_results]
        )

        w_adjusted = holm_adjust(
            [x["Wilcoxon_p"] for x in metric_results]
        )

        for i, result in enumerate(metric_results):
            result["Holm_t_p"] = t_adjusted[i]
            result["Holm_Wilcoxon_p"] = w_adjusted[i]
            summary_records.append(result)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # Direct paired comparison between V1 and V2.
    version_records = []

    for metric in METRIC_MAP:
        v1_values = v1_ppo[metric].reindex(common_seeds)

        v2_values = (
            v2[v2["Scheduler"] == "PPO"]
            .set_index("Seed")[metric]
            .reindex(common_seeds)
        )

        result = paired_statistics(
            v2_values,
            v1_values,
            rng
        )

        result["Metric"] = metric
        result["Comparison"] = "PPO V2 minus PPO V1"

        version_records.append(result)

    version_df = pd.DataFrame(version_records)
    version_df.to_csv(OUT_VERSION, index=False)

    print("\nV2 PAIRED STATISTICAL SUMMARY")
    print(summary_df.round(6).to_string(index=False))

    print("\nPPO V1 vs PPO V2")
    print(version_df.round(6).to_string(index=False))

    print("\nFILES SAVED:")
    print(OUT_RAW)
    print(OUT_SUMMARY)
    print(OUT_VERSION)

    print("\nSTATISTICAL VALIDATION COMPLETED")
    print("All comparisons use the same 20 held-out seeds.")
    print("Negative PPO-minus-baseline differences favor PPO V2")
    print("for these four lower-is-better metrics.")


if __name__ == "__main__":
    main()
