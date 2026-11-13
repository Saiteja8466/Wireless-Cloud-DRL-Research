
import os
import numpy as np
import pandas as pd

from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "results/ppo_heldout_raw_results.csv"

OUTPUT_RAW = "results/ppo_paired_statistical_raw.csv"
OUTPUT_SUMMARY = "results/ppo_paired_statistical_summary.csv"

BOOTSTRAP_RESAMPLES = 10000
RANDOM_SEED = 42

PPO_NAME = "PPO"

BASELINES = [
    "FCFS",
    "Min-Min Offline",
    "Online Earliest Finish",
    "Random",
    "Round Robin"
]

METRICS = [
    "Makespan_s",
    "Average_Response_s",
    "Deadline_Miss_Rate_pct",
    "Average_Energy_J"
]


# ============================================================
# HOLM MULTIPLE-COMPARISON CORRECTION
# ============================================================

def holm_adjust(p_values):

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):

        value = (n - rank) * p_values[index]

        running_max = max(running_max, value)

        adjusted[index] = min(running_max, 1.0)

    return adjusted


# ============================================================
# PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def paired_bootstrap_ci(differences, rng):

    differences = np.asarray(differences, dtype=float)

    n = len(differences)

    bootstrap_means = np.empty(BOOTSTRAP_RESAMPLES)

    for i in range(BOOTSTRAP_RESAMPLES):

        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(sample)

    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)

    return lower, upper


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "Seed",
        "Scheduler"
    ] + METRICS

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.duplicated(["Seed", "Scheduler"]).any():
        raise ValueError(
            "Duplicate Seed-Scheduler combinations found."
        )

    expected_schedulers = [PPO_NAME] + BASELINES

    for seed in sorted(df["Seed"].unique()):

        seed_data = df[df["Seed"] == seed]

        available = set(seed_data["Scheduler"])

        missing_schedulers = (
            set(expected_schedulers) - available
        )

        if missing_schedulers:
            raise ValueError(
                f"Seed {seed} is missing schedulers: "
                f"{missing_schedulers}"
            )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    print("Dataset validation successful.")
    print("Total records:", len(df))
    print("Unique test seeds:", df["Seed"].nunique())
    print("Schedulers:", sorted(df["Scheduler"].unique()))

    return df


# ============================================================
# PAIRED STATISTICAL ANALYSIS
# ============================================================

def run_analysis(df):

    rng = np.random.default_rng(RANDOM_SEED)

    raw_records = []
    summary_records = []

    for metric in METRICS:

        print("\nMetric:", metric)

        metric_results = []

        for baseline in BASELINES:

            pivot = df.pivot(
                index="Seed",
                columns="Scheduler",
                values=metric
            )

            paired = pivot[
                [PPO_NAME, baseline]
            ].dropna()

            ppo_values = paired[PPO_NAME].to_numpy()
            baseline_values = paired[baseline].to_numpy()

            # PPO minus baseline
            differences = ppo_values - baseline_values

            n = len(differences)

            mean_diff = np.mean(differences)
            median_diff = np.median(differences)
            std_diff = np.std(differences, ddof=1)

            # Paired t-test
            t_result = stats.ttest_rel(
                ppo_values,
                baseline_values
            )

            t_statistic = float(t_result.statistic)
            t_pvalue = float(t_result.pvalue)

            # Wilcoxon signed-rank test
            if np.allclose(differences, 0):

                wilcoxon_statistic = 0.0
                wilcoxon_pvalue = 1.0

            else:

                w_result = stats.wilcoxon(
                    ppo_values,
                    baseline_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )

                wilcoxon_statistic = float(
                    w_result.statistic
                )

                wilcoxon_pvalue = float(
                    w_result.pvalue
                )

            # Paired bootstrap confidence interval
            ci_lower, ci_upper = paired_bootstrap_ci(
                differences,
                rng
            )

            # Paired standardized effect size (Cohen's dz)
            if std_diff > 0:

                cohens_dz = mean_diff / std_diff

            else:

                cohens_dz = np.nan

            record = {
                "Metric": metric,
                "Comparison": f"PPO vs {baseline}",
                "N": n,
                "PPO_Mean": np.mean(ppo_values),
                "Baseline_Mean": np.mean(baseline_values),
                "Mean_Paired_Difference": mean_diff,
                "Median_Paired_Difference": median_diff,
                "Std_Paired_Difference": std_diff,
                "Bootstrap_95CI_Lower": ci_lower,
                "Bootstrap_95CI_Upper": ci_upper,
                "Cohens_dz": cohens_dz,
                "Paired_t_Statistic": t_statistic,
                "Paired_t_pvalue": t_pvalue,
                "Wilcoxon_Statistic": wilcoxon_statistic,
                "Wilcoxon_pvalue": wilcoxon_pvalue
            }

            metric_results.append(record)

            for seed, difference in zip(
                paired.index,
                differences
            ):

                raw_records.append({
                    "Seed": seed,
                    "Metric": metric,
                    "Baseline": baseline,
                    "PPO_Value": ppo_values[
                        list(paired.index).index(seed)
                    ],
                    "Baseline_Value": baseline_values[
                        list(paired.index).index(seed)
                    ],
                    "PPO_Minus_Baseline": difference
                })

        # Holm adjustment separately for each metric
        # and each statistical test family.

        t_adjusted = holm_adjust([
            item["Paired_t_pvalue"]
            for item in metric_results
        ])

        w_adjusted = holm_adjust([
            item["Wilcoxon_pvalue"]
            for item in metric_results
        ])

        for i, item in enumerate(metric_results):

            item["Paired_t_Holm_pvalue"] = t_adjusted[i]

            item["Wilcoxon_Holm_pvalue"] = w_adjusted[i]

            item["Significant_t_Holm_0.05"] = (
                t_adjusted[i] < 0.05
            )

            item["Significant_Wilcoxon_Holm_0.05"] = (
                w_adjusted[i] < 0.05
            )

            item["Direction"] = (
                "PPO better"
                if item["Mean_Paired_Difference"] < 0
                else "Baseline better"
                if item["Mean_Paired_Difference"] > 0
                else "No mean difference"
            )

            summary_records.append(item)

    raw_df = pd.DataFrame(raw_records)

    summary_df = pd.DataFrame(summary_records)

    os.makedirs("results", exist_ok=True)

    raw_df.to_csv(
        OUTPUT_RAW,
        index=False
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    return raw_df, summary_df


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(summary_df):

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 6)

    columns = [
        "Metric",
        "Comparison",
        "Mean_Paired_Difference",
        "Bootstrap_95CI_Lower",
        "Bootstrap_95CI_Upper",
        "Paired_t_Holm_pvalue",
        "Wilcoxon_Holm_pvalue",
        "Cohens_dz",
        "Direction"
    ]

    print("\n")
    print("=" * 110)
    print("PPO PAIRED STATISTICAL ANALYSIS")
    print("=" * 110)

    print(
        summary_df[columns].to_string(index=False)
    )

    print("\n")
    print("STATISTICAL SIGNIFICANCE: HOLM-ADJUSTED P < 0.05")

    print("\nFiles saved:")
    print(OUTPUT_RAW)
    print(OUTPUT_SUMMARY)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MILESTONE 7.7 - PPO STATISTICAL VALIDATION")
    print("=" * 65)

    df = load_data()

    raw_df, summary_df = run_analysis(df)

    display_results(summary_df)

    print("\nMILESTONE 7.7 ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
