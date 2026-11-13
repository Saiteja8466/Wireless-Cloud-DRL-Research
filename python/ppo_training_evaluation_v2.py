
import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
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


# --------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------

NUM_TASKS = 200
NUM_VMS = 5

TRAIN_SEED = 42
TEST_SEEDS = list(range(201, 221))

TOTAL_TIMESTEPS = 100000

MODEL_DIR = "models"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    "models",
    "wireless_cloud_ppo_v2"

)


# --------------------------------------------------
# ENVIRONMENT FACTORY
# --------------------------------------------------

def make_training_env():
    env = WirelessCloudEnv(
        num_tasks=NUM_TASKS,
        num_vms=NUM_VMS
    )

    return Monitor(env)


# --------------------------------------------------
# TRAIN PPO
# --------------------------------------------------

def train_ppo():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nPPO TRAINING")
    print("=" * 60)

    vec_env = DummyVecEnv([
        make_training_env
    ])

    vec_env.seed(TRAIN_SEED)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=TRAIN_SEED,
        device="cpu"
    )

    print("Training timesteps:", TOTAL_TIMESTEPS)
    print("Training seed:", TRAIN_SEED)
    print("Device: CPU")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False
    )

    model.save(MODEL_PATH)

    vec_env.close()

    print("\nPPO TRAINING COMPLETED")
    print("Model saved:", MODEL_PATH + ".zip")

    return model


# --------------------------------------------------
# EVALUATE PPO ON HELD-OUT WORKLOADS
# --------------------------------------------------

def evaluate_ppo(model, seed):

    env = WirelessCloudEnv(
        num_tasks=NUM_TASKS,
        num_vms=NUM_VMS
    )

    obs, reset_info = env.reset(seed=seed)

    completion_times = []
    response_times = []
    energy_values = []

    deadline_misses = 0
    task_count = 0

    while True:

        task = env.tasks[env.current_task]

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            int(action)
        )

        # Use the environment's reported completion and energy.
        if "completion_time" not in info or "energy_j" not in info:
            raise KeyError(
              "Expected 'completion_time' and 'energy_j' in environment info. "
               f"Available keys: {list(info.keys())}"
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

    env.close()

    if task_count != NUM_TASKS:
        raise RuntimeError(
            f"Expected {NUM_TASKS} tasks, evaluated {task_count}"
        )

    return {
        "Makespan_s": float(max(completion_times)),
        "Average_Response_s": float(np.mean(response_times)),
        "Deadline_Misses": int(deadline_misses),
        "Deadline_Miss_Rate_pct": (
            deadline_misses / task_count * 100.0
        ),
        "Average_Energy_J": float(np.mean(energy_values)),
        "Total_Energy_J": float(np.sum(energy_values))
    }


# --------------------------------------------------
# BASELINE EVALUATION ON IDENTICAL HELD-OUT SEEDS
# --------------------------------------------------

def evaluate_all_schedulers(model):

    raw_results = []

    print("\nHELD-OUT EVALUATION")
    print("=" * 60)
    print("Test seeds:", TEST_SEEDS)

    for run_number, seed in enumerate(TEST_SEEDS, start=1):

        tasks, vm_mips, vm_power = generate_workload(seed)

        schedulers = {
            "FCFS": lambda: schedule_fcfs(tasks, vm_mips),

            "Round Robin": lambda: schedule_round_robin(tasks),

            "Min-Min Offline": lambda: (
                schedule_min_min_offline(tasks, vm_mips)
            ),

            "Random": lambda: schedule_random(tasks, seed),

            "Online Earliest Finish": lambda: (
                schedule_online_earliest_finish(tasks, vm_mips)
            )
        }

        for scheduler_name, scheduler in schedulers.items():

            assignments = scheduler()

            metrics = calculate_metrics(
                tasks,
                assignments,
                vm_mips,
                vm_power
            )

            raw_results.append({
                "Seed": seed,
                "Scheduler": scheduler_name,
                **metrics
            })

        ppo_metrics = evaluate_ppo(model, seed)

        raw_results.append({
            "Seed": seed,
            "Scheduler": "PPO",
            **ppo_metrics
        })

        print(
            f"Completed test seed {seed} "
            f"({run_number}/{len(TEST_SEEDS)})"
        )

    return pd.DataFrame(raw_results)


# --------------------------------------------------
# SAVE RESULTS AND SUMMARY
# --------------------------------------------------

def save_results(raw_df):

    metric_columns = [
        "Makespan_s",
        "Average_Response_s",
        "Deadline_Misses",
        "Deadline_Miss_Rate_pct",
        "Average_Energy_J",
        "Total_Energy_J"
    ]

    raw_path = os.path.join(
        RESULTS_DIR,
        "ppo_heldout_raw_results.csv"
    )

    raw_df.to_csv(raw_path, index=False)

    summary = (
        raw_df.groupby("Scheduler")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "ppo_heldout_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nHELD-OUT PERFORMANCE SUMMARY")
    print("=" * 80)

    display_columns = [
        "Makespan_s",
        "Average_Response_s",
        "Deadline_Miss_Rate_pct",
        "Average_Energy_J"
    ]

    simple_summary = raw_df.groupby("Scheduler")[
        display_columns
    ].mean()

    print(
        simple_summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nFILES SAVED:")
    print(raw_path)
    print(summary_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nWIRELESS CLOUD PPO RESEARCH")
    print("=" * 60)

    model = train_ppo()
    print("Previously trained PPO model loaded successfully.")

    raw_df = evaluate_all_schedulers(model)

    save_results(raw_df)

    print("\nMILESTONE 7.6 COMPLETED")
    print("=" * 60)

    print("PPO model:", MODEL_PATH + ".zip")
    print("Held-out seeds:", len(TEST_SEEDS))

    print(
        "\nNOTE: Min-Min is an offline benchmark. "
        "All other schedulers use arrival-order decisions."
    )

    print(
        "Results are simulation-based, not real cloud measurements."
    )


if __name__ == "__main__":
    main()
