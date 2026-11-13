import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from wireless_cloud_ppo_env_v2 import WirelessCloudEnv

from multi_seed_validation import (
    generate_workload,
    schedule_fcfs,
    schedule_round_robin,
    schedule_random,
    schedule_min_min_offline,
    schedule_online_earliest_finish,
    calculate_metrics
)

NUM_TASKS = 200
NUM_VMS = 5
TEST_SEEDS = list(range(201, 221))
TRAINING_SEEDS = [7, 42, 99, 123, 2026]

MODEL_DIR = "models/ppo_logged"
OUTPUT_DIR = "results/ppo_logged_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_ppo(model, seed):
    env = WirelessCloudEnv(
        num_tasks=NUM_TASKS,
        num_vms=NUM_VMS
    )

    obs, _ = env.reset(seed=seed)

    completion_times = []
    response_times = []
    energy_values = []
    deadline_misses = 0
    task_count = 0

    try:
        while True:
            task = env.tasks[env.current_task]

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                int(action)
            )

            if "completion_time" not in info or "energy_j" not in info:
                raise KeyError(
                    f"Missing expected environment metrics: {list(info.keys())}"
                )

            completion = float(info["completion_time"])
            energy = float(info["energy_j"])
            response = completion - float(task["arrival"])
            missed = response > float(task["deadline"])

            completion_times.append(completion)
            response_times.append(response)
            energy_values.append(energy)

            deadline_misses += int(missed)
            task_count += 1

            if terminated or truncated:
                break

    finally:
        env.close()

    if task_count != NUM_TASKS:
        raise RuntimeError(
            f"Expected {NUM_TASKS} tasks, evaluated {task_count}"
        )

    return {
        "Makespan_s": float(max(completion_times)),
        "Average_Response_s": float(np.mean(response_times)),
        "Deadline_Misses": int(deadline_misses),
        "Deadline_Miss_Rate_pct": deadline_misses / task_count * 100.0,
        "Average_Energy_J": float(np.mean(energy_values)),
        "Total_Energy_J": float(np.sum(energy_values))
    }


def main():
    print("=" * 70)
    print("FINAL LOGGED PPO MULTI-SEED EVALUATION")
    print("=" * 70)

    # Load all five previously trained models.
    models = {}

    for training_seed in TRAINING_SEEDS:
        model_path = os.path.join(
            MODEL_DIR,
            f"wireless_cloud_ppo_logged_seed_{training_seed}.zip"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing trained model: {model_path}"
            )

        models[training_seed] = PPO.load(
            model_path,
            device="cpu"
        )

        print(f"Loaded PPO model: seed {training_seed}")

    raw_results = []

    for run_number, test_seed in enumerate(TEST_SEEDS, start=1):
        print(f"\nEvaluating test seed {test_seed} ({run_number}/20)")

        # Evaluate existing non-PPO baselines on this workload.
        tasks, vm_mips, vm_power = generate_workload(test_seed)

        baseline_schedulers = {
            "FCFS": lambda: schedule_fcfs(tasks, vm_mips),
            "Round Robin": lambda: schedule_round_robin(tasks),
            "Min-Min Offline": lambda: schedule_min_min_offline(
                tasks, vm_mips
            ),
            "Random": lambda: schedule_random(tasks, test_seed),
            "Online Earliest Finish": lambda: (
                schedule_online_earliest_finish(tasks, vm_mips)
            )
        }

        for scheduler_name, scheduler in baseline_schedulers.items():
            assignments = scheduler()

            metrics = calculate_metrics(
                tasks,
                assignments,
                vm_mips,
                vm_power
            )

            raw_results.append({
                "Seed": test_seed,
                "Scheduler": scheduler_name,
                "Training_Seed": None,
                **metrics
            })

        # Evaluate every logged PPO model on the same test workload.
        for training_seed, model in models.items():
            metrics = evaluate_ppo(model, test_seed)

            raw_results.append({
                "Seed": test_seed,
                "Scheduler": f"PPO",
                "Training_Seed": training_seed,
                **metrics
            })

    raw_df = pd.DataFrame(raw_results)

    raw_path = os.path.join(
        OUTPUT_DIR,
        "logged_ppo_final_raw_results.csv"
    )
    raw_df.to_csv(raw_path, index=False)

    metric_columns = [
        "Makespan_s",
        "Average_Response_s",
        "Deadline_Misses",
        "Deadline_Miss_Rate_pct",
        "Average_Energy_J",
        "Total_Energy_J"
    ]

    summary = (
        raw_df.groupby(["Scheduler", "Training_Seed"], dropna=False)
        [metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "logged_ppo_final_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    simple_summary = (
        raw_df.groupby(["Scheduler", "Training_Seed"], dropna=False)
        [["Makespan_s", "Average_Response_s",
          "Deadline_Miss_Rate_pct", "Average_Energy_J"]]
        .mean()
        .reset_index()
    )

    print("\nFINAL EVALUATION SUMMARY")
    print("=" * 90)
    print(simple_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTotal evaluation records:", len(raw_df))
    print("Expected evaluation records:", 200)
    print("Raw results:", raw_path)
    print("Summary:", summary_path)

    if len(raw_df) != 200:
        raise RuntimeError("Unexpected evaluation record count.")

    print("\nMILESTONE 9 EVALUATION COMPLETED")
    print("All original training models and results were preserved.")


if __name__ == "__main__":
    main()
