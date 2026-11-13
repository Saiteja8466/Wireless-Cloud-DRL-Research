import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "results/ppo_logged"
GRAPH_DIR = "graphs/ppo_training_stability"
os.makedirs(GRAPH_DIR, exist_ok=True)

metrics = pd.read_csv(
    os.path.join(DATA_DIR, "ppo_tensorboard_metrics.csv")
)
episodes = pd.read_csv(
    os.path.join(DATA_DIR, "ppo_episode_monitor_data.csv")
)

# 1. Episode reward
plt.figure()
for seed, group in metrics[
    metrics["Metric"] == "rollout/ep_rew_mean"
].groupby("Training_Seed"):
    plt.plot(group["Step"], group["Value"], label=f"Seed {seed}")
plt.xlabel("Training timesteps")
plt.ylabel("Mean episode reward")
plt.title("PPO Training Reward Across Five Seeds")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "01_reward_curve.png"), dpi=300)
plt.close()

# 2. Explained variance
plt.figure()
for seed, group in metrics[
    metrics["Metric"] == "train/explained_variance"
].groupby("Training_Seed"):
    plt.plot(group["Step"], group["Value"], label=f"Seed {seed}")
plt.xlabel("Training timesteps")
plt.ylabel("Explained variance")
plt.title("PPO Value Function Explained Variance")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "02_explained_variance.png"), dpi=300)
plt.close()

# 3. Value loss
plt.figure()
for seed, group in metrics[
    metrics["Metric"] == "train/value_loss"
].groupby("Training_Seed"):
    plt.plot(group["Step"], group["Value"], label=f"Seed {seed}")
plt.xlabel("Training timesteps")
plt.ylabel("Value loss")
plt.title("PPO Value Loss Across Five Seeds")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "03_value_loss.png"), dpi=300)
plt.close()

# 4. Approximate KL divergence
plt.figure()
for seed, group in metrics[
    metrics["Metric"] == "train/approx_kl"
].groupby("Training_Seed"):
    plt.plot(group["Step"], group["Value"], label=f"Seed {seed}")
plt.xlabel("Training timesteps")
plt.ylabel("Approximate KL")
plt.title("PPO Approximate KL Divergence")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "04_approx_kl.png"), dpi=300)
plt.close()

# 5. Policy gradient loss
plt.figure()
for seed, group in metrics[
    metrics["Metric"] == "train/policy_gradient_loss"
].groupby("Training_Seed"):
    plt.plot(group["Step"], group["Value"], label=f"Seed {seed}")
plt.xlabel("Training timesteps")
plt.ylabel("Policy gradient loss")
plt.title("PPO Policy Gradient Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "05_policy_gradient_loss.png"), dpi=300)
plt.close()

# 6. Episode return from Monitor logs
plt.figure()
for seed, group in episodes.groupby("Training_Seed"):
    group = group.sort_values("t")
    plt.plot(group["t"], group["r"], label=f"Seed {seed}", alpha=0.8)
plt.xlabel("Elapsed training time (seconds)")
plt.ylabel("Episode return")
plt.title("Monitor Episode Returns Across Five Seeds")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GRAPH_DIR, "06_monitor_episode_returns.png"), dpi=300)
plt.close()

# Training summary
summary = episodes.groupby("Training_Seed").agg(
    Episodes=("r", "count"),
    Mean_Episode_Reward=("r", "mean"),
    Std_Episode_Reward=("r", "std"),
    Mean_Episode_Length=("l", "mean"),
    Final_Episode_Reward=("r", "last")
).reset_index()

summary.to_csv(
    os.path.join(DATA_DIR, "ppo_episode_training_summary.csv"),
    index=False
)

print("\nGRAPH GENERATION COMPLETED")
print(summary.to_string(index=False))
print("\nGraphs saved in:", GRAPH_DIR)
print("Summary saved in:", DATA_DIR)

