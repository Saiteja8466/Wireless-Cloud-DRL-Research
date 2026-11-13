
from stable_baselines3.common.env_checker import check_env
from wireless_cloud_ppo_env import WirelessCloudEnv

env = WirelessCloudEnv(num_tasks=200, num_vms=5)

print("Checking Gymnasium environment...")
check_env(env)

print("\nEnvironment check passed!")

obs, info = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

total_reward = 0.0
deadline_misses = 0
total_energy = 0.0

for step in range(200):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    deadline_misses += int(info["deadline_missed"])
    total_energy += info["energy_j"]

    if step < 5:
        print(
            f"Task {step + 1}: "
            f"VM={info['vm_selected']}, "
            f"SNR={info['snr_db']:.2f} dB, "
            f"Response={info['response_time']:.4f} s, "
            f"Energy={info['energy_j']:.2f} J"
        )

    if terminated or truncated:
        break

print("\n===== RANDOM SCHEDULER TEST =====")
print("Tasks completed:", step + 1)
print("Total reward:", round(total_reward, 4))
print("Deadline misses:", deadline_misses)
print("Average energy:", round(total_energy / (step + 1), 4), "J")

env.close()
