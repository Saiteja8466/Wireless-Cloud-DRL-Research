import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = "training_logs"
OUTPUT_DIR = "results/ppo_logged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_metrics = []

for seed in [7, 42, 99, 123, 2026]:
    run_dir = os.path.join(LOG_ROOT, f"ppo_seed_{seed}")
    event_files = glob.glob(
        os.path.join(run_dir, "**", "events.out.tfevents.*"),
        recursive=True
    )

    if not event_files:
        print(f"No TensorBoard event file found for seed {seed}")
        continue

    for event_file in event_files:
        accumulator = EventAccumulator(event_file)
        accumulator.Reload()

        available_tags = accumulator.Tags().get("scalars", [])
        print(f"\nSeed {seed} - Available metrics:")
        print(available_tags)

        for tag in available_tags:
            for item in accumulator.Scalars(tag):
                all_metrics.append({
                    "Training_Seed": seed,
                    "Metric": tag,
                    "Step": item.step,
                    "Value": item.value
                })

metrics_df = pd.DataFrame(all_metrics)

if metrics_df.empty:
    raise RuntimeError("No scalar metrics were extracted.")

output_file = os.path.join(
    OUTPUT_DIR, "ppo_tensorboard_metrics.csv"
)
metrics_df.to_csv(output_file, index=False)

print("\nTensorBoard extraction completed.")
print("Total records:", len(metrics_df))
print("Metrics:", sorted(metrics_df["Metric"].unique()))
print("Saved:", output_file)

monitor_frames = []

for seed in [7, 42, 99, 123, 2026]:
    monitor_file = os.path.join(
        LOG_ROOT, f"ppo_seed_{seed}", "monitor.csv"
    )
    if os.path.exists(monitor_file):
        df = pd.read_csv(monitor_file, skiprows=1)
        df["Training_Seed"] = seed
        monitor_frames.append(df)

if monitor_frames:
    monitor_df = pd.concat(monitor_frames, ignore_index=True)
    monitor_output = os.path.join(
        OUTPUT_DIR, "ppo_episode_monitor_data.csv"
    )
    monitor_df.to_csv(monitor_output, index=False)
    print("Episode records:", len(monitor_df))
    print("Saved:", monitor_output)

