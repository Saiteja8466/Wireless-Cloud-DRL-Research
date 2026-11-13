
import os
import numpy as np
import pandas as pd

from scipy import stats


INPUT_FILE = "results/multi_seed_raw_results.csv"
OUTPUT_DIR = "results"

REFERENCE = "Online Earliest Finish"

COMPARATORS = [
    "FCFS",
    "Round Robin",
    "Min-Min Offline",
    "Random"
]

METRICS = [
    "Makespan_s",
    "Average_Response_s",
    "Deadline_Miss_Rate_pct",
    "Average_Energy_J"
]

BOOTSTRAP_SAMPLES = 10000
SEED = 2026


def holm_adjust(p_values):
    """Holm correction for multiple statistical comparisons."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)

    adjusted = np.empty(len(p_values))
    running_max = 0.0
    count = len(p_values)

    for rank, index in enumerate(order):
        value = (count - rank) * p_values[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def bootstrap_mean_ci(values, rng):
    """Percentile bootstrap CI for paired mean differences."""

    values = np.asarray(values, dtype=float)
    n = len(values)

    samples = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, n),
        replace=True
    )

    bootstrap_means = samples.mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return float(lower), float(upper)


def analyze():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Please check the results folder."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Seed",
        "Scheduler",
        *METRICS
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in raw CSV: {missing}"
        )

    # Confirm one row per seed and scheduler.
    assert not df.duplicated(
        subset=["Seed", "Scheduler"]
    ).any()

    rng = np.random.default_rng(SEED)

    results = []

    for comparator in COMPARATORS:
        for metric in METRICS:

            reference_data = df[
                df["Scheduler"] == REFERENCE
            ][["Seed", metric]]

            comparator_data = df[
                df["Scheduler"] == comparator
            ][["Seed", metric]]

            paired = reference_data.merge(
                comparator_data,
                on="Seed",
                suffixes=("_reference", "_comparator"),
                validate="one_to_one"
            )

            if len(paired) != 20:
                raise ValueError(
                    f"Expected 20 paired seeds for "
                    f"{comparator} and {metric}; "
                    f"found {len(paired)}"
                )

            # Negative difference favors the reference
            # for all four metrics.
            differences = (
                paired[f"{metric}_reference"]
                - paired[f"{metric}_comparator"]
            ).to_numpy(dtype=float)

            mean_difference = float(np.mean(differences))
            std_difference = float(
                np.std(differences, ddof=1)
            )

            # Paired t-test.
            t_result = stats.ttest_1samp(
                differences,
                popmean=0.0
            )

            # Wilcoxon signed-rank test.
            if np.allclose(differences, 0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(
                    stats.wilcoxon(
                        differences,
                        alternative="two-sided",
                        method="auto"
                    ).pvalue
                )

            # Paired bootstrap confidence interval.
            ci_lower, ci_upper = bootstrap_mean_ci(
                differences,
                rng
            )

            results.append({
                "Reference": REFERENCE,
                "Comparator": comparator,
                "Metric": metric,
                "Paired_Seeds": len(differences),
                "Reference_Mean": float(
                    paired[f"{metric}_reference"].mean()
                ),
                "Comparator_Mean": float(
                    paired[f"{metric}_comparator"].mean()
                ),
                "Mean_Paired_Difference": mean_difference,
                "Difference_Std": std_difference,
                "Bootstrap_95CI_Lower": ci_lower,
                "Bootstrap_95CI_Upper": ci_upper,
                "Paired_t_p_value": float(t_result.pvalue),
                "Wilcoxon_p_value": wilcoxon_p
            })

    result_df = pd.DataFrame(results)

    # Holm correction separately for each metric and test.
    result_df["Paired_t_Holm_p"] = np.nan
    result_df["Wilcoxon_Holm_p"] = np.nan

    for metric in METRICS:
        mask = result_df["Metric"] == metric

        result_df.loc[mask, "Paired_t_Holm_p"] = (
            holm_adjust(
                result_df.loc[
                    mask, "Paired_t_p_value"
                ].to_numpy()
            )
        )

        result_df.loc[mask, "Wilcoxon_Holm_p"] = (
            holm_adjust(
                result_df.loc[
                    mask, "Wilcoxon_p_value"
                ].to_numpy()
            )
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "paired_statistical_comparison.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\nPAIRED STATISTICAL COMPARISON")
    print("=" * 75)

    for _, row in result_df.iterrows():
        print(
            f"\n{row['Metric']} | "
            f"{REFERENCE} vs {row['Comparator']}"
        )

        print(
            "Mean paired difference:",
            f"{row['Mean_Paired_Difference']:.6f}"
        )

        print(
            "Bootstrap 95% CI:",
            f"[{row['Bootstrap_95CI_Lower']:.6f}, "
            f"{row['Bootstrap_95CI_Upper']:.6f}]"
        )

        print(
            "Paired t-test p:",
            f"{row['Paired_t_p_value']:.6f}",
            "| Holm adjusted:",
            f"{row['Paired_t_Holm_p']:.6f}"
        )

        print(
            "Wilcoxon p:",
            f"{row['Wilcoxon_p_value']:.6f}",
            "| Holm adjusted:",
            f"{row['Wilcoxon_Holm_p']:.6f}"
        )

    print("\nSaved:", output_path)
    print("\nPAIRED ANALYSIS COMPLETED")


if __name__ == "__main__":
    analyze()
