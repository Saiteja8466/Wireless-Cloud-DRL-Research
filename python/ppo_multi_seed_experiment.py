import os
import pandas as pd

import ppo_training_evaluation_v2 as base


# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

TRAINING_SEEDS = [42, 123, 2026, 7, 99]
TEST_SEEDS = list(range(201, 221))

TOTAL_TIMESTEPS = 100000

RAW_FILE = "results/ppo_v2_multiseed_raw_results.csv"
SEED_SUMMARY_FILE = "results/ppo_v2_multiseed_summary.csv"
OVERALL_FILE = "results/ppo_v2_multiseed_overall_summary.csv"


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Protect existing experiment results.
    for path in [RAW_FILE, SEED_SUMMARY_FILE, OVERALL_FILE]:
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists. Rename or back it up "
                "before starting a new experiment."
            )

    base.TEST_SEEDS = TEST_SEEDS
    base.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS

    all_results = []

    print("=" * 70)
    print("PPO V2 MULTI-SEED TRAINING AND ROBUSTNESS EXPERIMENT")
    print("=" * 70)
    print("Training seeds:", TRAINING_SEEDS)
    print("Test seeds:", TEST_SEEDS)
    print("Timesteps per model:", TOTAL_TIMESTEPS)
    print("Total planned evaluation episodes:",
          len(TRAINING_SEEDS) * len(TEST_SEEDS))

    for run_number, train_seed in enumerate(TRAINING_SEEDS, start=1):

        print("\n" + "=" * 70)
        print(f"TRAINING RUN {run_number}/{len(TRAINING_SEEDS)}")
        print("Training seed:", train_seed)
        print("=" * 70)

        # Give every training run a separate model filename.
        base.TRAIN_SEED = train_seed
        base.MODEL_PATH = os.path.join(
            "models",
            f"wireless_cloud_ppo_v2_trainseed_{train_seed}"
        )

        # Train a fresh PPO V2 model.
        model = base.train_ppo()

        print("\nEvaluating trained model on identical held-out workloads...")

        seed_results = []

        for test_number, test_seed in enumerate(TEST_SEEDS, start=1):

            metrics = base.evaluate_ppo(model, test_seed)

            row = {
                "Training_Seed": train_seed,
                "Seed": test_seed,
                "Scheduler": "PPO",
                **metrics
            }

            seed_results.append(row)

            print(
                f"Training seed {train_seed} | "
                f"Test seed {test_seed} | "
                f"{test_number}/{len(TEST_SEEDS)} completed"
            )

        all_results.extend(seed_results)

        # Save after each training run to protect completed results.
        raw_df = pd.DataFrame(all_results)
        raw_df.to_csv(RAW_FILE, index=False)

        print("\nTraining run completed:", train_seed)
        print("Model:", base.MODEL_PATH + ".zip")
        print("Raw results saved:", RAW_FILE)

    # --------------------------------------------------
    # FINAL SUMMARIES
    # --------------------------------------------------

    raw_df = pd.DataFrame(all_results)

    metrics = [
        "Makespan_s",
        "Average_Response_s",
        "Deadline_Misses",
        "Deadline_Miss_Rate_pct",
        "Average_Energy_J",
        "Total_Energy_J"
    ]

    seed_summary = (
        raw_df.groupby("Training_Seed")[metrics]
        .agg(["mean", "std"])
    )

    seed_summary.to_csv(SEED_SUMMARY_FILE)

    overall_summary = (
        raw_df[metrics]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(columns={"index": "Metric"})
    )

    overall_summary.to_csv(OVERALL_FILE, index=False)

    print("\n" + "=" * 70)
    print("MULTI-SEED EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nPER-TRAINING-SEED PERFORMANCE")
    print(seed_summary.round(5).to_string())

    print("\nOVERALL PERFORMANCE ACROSS ALL EVALUATIONS")
    print(overall_summary.round(5).to_string(index=False))

    print("\nFILES SAVED:")
    print(RAW_FILE)
    print(SEED_SUMMARY_FILE)
    print(OVERALL_FILE)

    print("\nAll five independently trained PPO models are preserved.")


if __name__ == "__main__":
    main()
