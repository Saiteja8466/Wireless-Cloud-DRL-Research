import os
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from wireless_cloud_ppo_env_v2 import WirelessCloudEnv


# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

TRAINING_SEEDS = [7, 42, 99, 123, 2026]

TOTAL_TIMESTEPS = 100000
CHECKPOINT_FREQ = 25000

NUM_TASKS = 200
NUM_VMS = 5

LOG_ROOT = "training_logs"
MODEL_ROOT = "models/ppo_logged"
RESULT_ROOT = "results/ppo_logged"


# --------------------------------------------------
# DIRECTORY SETUP
# --------------------------------------------------

for directory in [LOG_ROOT, MODEL_ROOT, RESULT_ROOT]:
    os.makedirs(directory, exist_ok=True)


# --------------------------------------------------
# ENVIRONMENT FACTORY
# --------------------------------------------------

def make_training_env(seed, monitor_file):

    def factory():
        env = WirelessCloudEnv(
            num_tasks=NUM_TASKS,
            num_vms=NUM_VMS
        )

        return Monitor(
            env,
            filename=monitor_file
        )

    return factory


# --------------------------------------------------
# TRAIN ONE PPO MODEL
# --------------------------------------------------

def train_one_seed(seed):

    run_name = f"ppo_seed_{seed}"

    log_dir = os.path.join(LOG_ROOT, run_name)
    model_dir = os.path.join(MODEL_ROOT, run_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    monitor_file = os.path.join(log_dir, "monitor.csv")

    final_model = os.path.join(
        MODEL_ROOT,
        f"wireless_cloud_ppo_logged_seed_{seed}"
    )

    if os.path.exists(final_model + ".zip"):
        raise FileExistsError(
            f"Final model already exists: {final_model}.zip"
        )

    if os.path.exists(monitor_file):
        raise FileExistsError(
            f"Monitor log already exists: {monitor_file}. "
            "Do not overwrite previous experiment logs."
        )

    print("\n" + "=" * 70)
    print(f"STARTING PPO TRAINING — SEED {seed}")
    print("=" * 70)

    vec_env = DummyVecEnv([
        make_training_env(seed, monitor_file)
    ])

    vec_env.seed(seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=checkpoint_dir,
        name_prefix=run_name
    )

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
        seed=seed,
        device="cpu",
        tensorboard_log=log_dir
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        log_interval=1
    )

    model.save(final_model)

    vec_env.close()

    print(f"\nTRAINING COMPLETED — SEED {seed}")
    print("Final model:", final_model + ".zip")
    print("Monitor log:", monitor_file)
    print("TensorBoard directory:", log_dir)

    return {
        "Training_Seed": seed,
        "Requested_Timesteps": TOTAL_TIMESTEPS,
        "Actual_Timesteps": model.num_timesteps,
        "Model_Path": final_model + ".zip",
        "Monitor_Path": monitor_file,
        "TensorBoard_Path": log_dir
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("\nLOGGED PPO MULTI-SEED EXPERIMENT")
    print("=" * 70)

    experiment_records = []

    for seed in TRAINING_SEEDS:

        record = train_one_seed(seed)
        experiment_records.append(record)

        manifest = pd.DataFrame(experiment_records)

        manifest.to_csv(
            os.path.join(
                RESULT_ROOT,
                "logged_ppo_training_manifest.csv"
            ),
            index=False
        )

    print("\n" + "=" * 70)
    print("ALL LOGGED PPO TRAINING RUNS COMPLETED")
    print("=" * 70)

    print(pd.DataFrame(experiment_records).to_string(index=False))


if __name__ == "__main__":
    main()
