
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wireless_cloud_ppo_env import WirelessCloudEnv


SEED = 42
NUM_TASKS = 200
NUM_VMS = 5

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_common_workload():
    """Generate one reproducible workload shared by all schedulers."""

    env = WirelessCloudEnv(
        num_tasks=NUM_TASKS,
        num_vms=NUM_VMS
    )

    env.reset(seed=SEED)

    tasks = [task.copy() for task in env.tasks]

    vm_mips = env.vm_mips.copy()
    vm_power = env.vm_power.copy()

    bandwidth = env.bandwidth_hz
    radio_power = env.radio_power_w

    # Shared FIFO wireless transmission queue.
    tx_available = 0.0

    for task in tasks:
        snr_linear = 10 ** (task["snr_db"] / 10.0)
        rate = bandwidth * np.log2(1.0 + snr_linear)

        tx_time = task["size_mb"] * 8e6 / rate

        tx_start = max(task["arrival"], tx_available)
        tx_finish = tx_start + tx_time

        task["tx_start"] = tx_start
        task["tx_finish"] = tx_finish
        task["tx_time"] = tx_time
        task["tx_energy"] = radio_power * tx_time

        tx_available = tx_finish

    return tasks, vm_mips, vm_power


def calculate_metrics(tasks, assignments, vm_mips, vm_power):
    """Calculate metrics from completed VM assignments."""

    vm_available = np.zeros(NUM_VMS, dtype=float)
    completion_times = []
    response_times = []
    energies = []
    deadline_misses = 0

    for task_index, vm in assignments:
        task = tasks[task_index]

        execution_time = task["cpu_mi"] / vm_mips[vm]

        execution_start = max(
            task["tx_finish"],
            vm_available[vm]
        )

        completion = execution_start + execution_time
        vm_available[vm] = completion

        response = completion - task["arrival"]

        execution_energy = vm_power[vm] * execution_time
        total_energy = task["tx_energy"] + execution_energy

        missed = response > task["deadline"]
        deadline_misses += int(missed)

        completion_times.append(completion)
        response_times.append(response)
        energies.append(total_energy)

    makespan = max(completion_times) if completion_times else 0.0

    return {
        "Makespan_s": makespan,
        "Average_Response_s": float(np.mean(response_times)),
        "Deadline_Misses": deadline_misses,
        "Deadline_Miss_Rate_pct": (
            deadline_misses / len(tasks) * 100.0
        ),
        "Average_Energy_J": float(np.mean(energies)),
        "Total_Energy_J": float(np.sum(energies))
    }


def run_fcfs(tasks):
    """FIFO task order; assign each task to earliest available VM."""

    vm_available = np.zeros(NUM_VMS, dtype=float)
    assignments = []

    for i, task in enumerate(tasks):
        vm = int(np.argmin(vm_available))
        execution_time = task["cpu_mi"] / VM_MIPS[vm]

        start = max(task["tx_finish"], vm_available[vm])
        vm_available[vm] = start + execution_time

        assignments.append((i, vm))

    return assignments


def run_round_robin(tasks):
    """Assign tasks cyclically to VMs in arrival order."""

    assignments = []

    for i in range(len(tasks)):
        vm = i % NUM_VMS
        assignments.append((i, vm))

    return assignments


def run_min_min(tasks):
    """
    Offline Min-Min baseline.

    All task transmission completion times are known.
    At each iteration, select the task-VM pair with the
    smallest predicted completion time.

    This is a clairvoyant benchmark, not an online policy.
    """

    vm_available = np.zeros(NUM_VMS, dtype=float)
    unscheduled = set(range(len(tasks)))
    assignments = []

    while unscheduled:
        best_task = None
        best_vm = None
        best_completion = float("inf")

        for task_index in unscheduled:
            task = tasks[task_index]

            for vm in range(NUM_VMS):
                execution_time = (
                    task["cpu_mi"] / VM_MIPS[vm]
                )

                start = max(
                    task["tx_finish"],
                    vm_available[vm]
                )

                completion = start + execution_time

                if completion < best_completion:
                    best_completion = completion
                    best_task = task_index
                    best_vm = vm

        assignments.append((best_task, best_vm))

        task = tasks[best_task]
        execution_time = task["cpu_mi"] / VM_MIPS[best_vm]

        start = max(
            task["tx_finish"],
            vm_available[best_vm]
        )

        vm_available[best_vm] = start + execution_time
        unscheduled.remove(best_task)

    return assignments


def run_random(tasks):
    """Random VM assignment with a fixed seed."""

    rng = np.random.default_rng(SEED)
    assignments = []

    for i in range(len(tasks)):
        vm = int(rng.integers(0, NUM_VMS))
        assignments.append((i, vm))

    return assignments


# Generate one shared workload.
tasks, VM_MIPS, VM_POWER = generate_common_workload()

schedulers = {
    "FCFS": run_fcfs,
    "Round Robin": run_round_robin,
    "Min-Min (Offline)": run_min_min,
    "Random": run_random
}

results = []

print("\nWIRELESS-CLOUD BASELINE COMPARISON")
print("=" * 65)
print("Tasks:", NUM_TASKS)
print("VMs:", NUM_VMS)
print("Random seed:", SEED)

for name, scheduler in schedulers.items():
    assignments = scheduler(tasks)

    metrics = calculate_metrics(
        tasks,
        assignments,
        VM_MIPS,
        VM_POWER
    )

    metrics["Scheduler"] = name
    results.append(metrics)

    print(f"\n{name}")
    for key, value in metrics.items():
        if key != "Scheduler":
            print(f"{key}: {value:.4f}")

df = pd.DataFrame(results)

columns = [
    "Scheduler",
    "Makespan_s",
    "Average_Response_s",
    "Deadline_Misses",
    "Deadline_Miss_Rate_pct",
    "Average_Energy_J",
    "Total_Energy_J"
]

df = df[columns]

csv_path = os.path.join(
    OUTPUT_DIR,
    "baseline_scheduler_comparison.csv"
)

df.to_csv(csv_path, index=False)

# Plot deadline miss rate.
plt.figure(figsize=(9, 5))
plt.bar(df["Scheduler"], df["Deadline_Miss_Rate_pct"])
plt.ylabel("Deadline miss rate (%)")
plt.title("Wireless-Cloud Scheduler Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "baseline_deadline_miss.png"),
    dpi=300
)
plt.close()

# Plot average response time.
plt.figure(figsize=(9, 5))
plt.bar(df["Scheduler"], df["Average_Response_s"])
plt.ylabel("Average response time (s)")
plt.title("Average Task Response Time")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "baseline_response_time.png"),
    dpi=300
)
plt.close()

# Plot average energy.
plt.figure(figsize=(9, 5))
plt.bar(df["Scheduler"], df["Average_Energy_J"])
plt.ylabel("Average energy (J)")
plt.title("Average Energy per Task")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "baseline_energy.png"),
    dpi=300
)
plt.close()

print("\n" + "=" * 65)
print("BASELINE COMPARISON COMPLETED")
print("=" * 65)

print(df.to_string(index=False))

print("\nSaved:")
print(csv_path)
print("results/baseline_deadline_miss.png")
print("results/baseline_response_time.png")
print("results/baseline_energy.png")

print(
    "\nNOTE: Min-Min uses advance knowledge of the complete "
    "workload. Treat it as an offline benchmark."
)
