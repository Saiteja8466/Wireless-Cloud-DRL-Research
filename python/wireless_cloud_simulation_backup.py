
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# -------------------------------
# Simulation configuration
# -------------------------------

NUM_TASKS = 500
NUM_VMS = 5
BANDWIDTH_HZ = 10e6
TRANSMISSION_POWER_W = 0.5

# Generate cloud tasks
tasks = pd.DataFrame({
    "task_id": np.arange(NUM_TASKS),
    "instructions_mi": np.random.randint(500, 10000, NUM_TASKS),
    "input_size_mb": np.random.randint(10, 500, NUM_TASKS),
    "deadline": np.random.randint(5, 50, NUM_TASKS),
    "snr_db": np.random.uniform(0, 30, NUM_TASKS)
})

# Virtual machines
vms = pd.DataFrame({
    "vm_id": np.arange(NUM_VMS),
    "mips": [1000, 1500, 2000, 2500, 3000],
    "power_w": [50, 65, 80, 95, 110]
})


# -------------------------------
# Wireless transmission model
# -------------------------------

def calculate_transmission_times(tasks):

    channel_available = 0.0
    transmission_records = []

    for _, task in tasks.iterrows():

        snr_linear = 10 ** (task.snr_db / 10)

        # Shannon channel capacity
        data_rate = BANDWIDTH_HZ * np.log2(1 + snr_linear)

        data_bits = task.input_size_mb * 8e6

        transmission_duration = data_bits / data_rate

        start_time = channel_available
        finish_time = start_time + transmission_duration

        channel_available = finish_time

        transmission_records.append({
            "task_id": int(task.task_id),
            "tx_start": start_time,
            "tx_finish": finish_time,
            "tx_duration": transmission_duration
        })

    return pd.DataFrame(transmission_records)


# -------------------------------
# Scheduling algorithms
# -------------------------------

def fcfs_schedule(tasks, vms):

    available = np.zeros(len(vms))
    assignments = {}

    for _, task in tasks.iterrows():

        vm_index = int(np.argmin(available))

        assignments[int(task.task_id)] = vm_index

        available[vm_index] += (
            task.instructions_mi / vms.iloc[vm_index].mips
        )

    return assignments


def round_robin_schedule(tasks, vms):

    assignments = {}

    for position, task_id in enumerate(tasks.task_id):
        assignments[int(task_id)] = position % len(vms)

    return assignments


def min_min_schedule(tasks, vms):

    remaining = tasks.copy()
    available = np.zeros(len(vms))
    assignments = {}

    while not remaining.empty:

        best_task = None
        best_vm = None
        best_finish = float("inf")

        for _, task in remaining.iterrows():

            for vm_index, vm in vms.iterrows():

                execution = task.instructions_mi / vm.mips

                finish = available[vm_index] + execution

                if finish < best_finish:
                    best_finish = finish
                    best_task = int(task.task_id)
                    best_vm = vm_index

        assignments[best_task] = best_vm

        task = remaining[
            remaining.task_id == best_task
        ].iloc[0]

        available[best_vm] += (
            task.instructions_mi / vms.iloc[best_vm].mips
        )

        remaining = remaining[
            remaining.task_id != best_task
        ]

    return assignments


# -------------------------------
# End-to-end evaluation
# -------------------------------

def evaluate_schedule(tasks, vms, assignments, tx_records):

    vm_available = np.zeros(len(vms))

    completion = {}
    execution_records = []

    total_energy = 0.0

    tx_lookup = tx_records.set_index("task_id")

    for _, task in tasks.iterrows():

        task_id = int(task.task_id)
        vm_index = assignments[task_id]

        vm = vms.iloc[vm_index]

        tx_finish = tx_lookup.loc[task_id, "tx_finish"]
        tx_duration = tx_lookup.loc[task_id, "tx_duration"]

        execution_time = task.instructions_mi / vm.mips

        execution_start = max(
            tx_finish,
            vm_available[vm_index]
        )

        execution_finish = execution_start + execution_time

        vm_available[vm_index] = execution_finish

        completion[task_id] = execution_finish

        transmission_energy = (
            TRANSMISSION_POWER_W * tx_duration
        )

        processing_energy = (
            vm.power_w * execution_time
        )

        total_energy += (
            transmission_energy + processing_energy
        )

        execution_records.append({
            "task_id": task_id,
            "tx_finish": tx_finish,
            "execution_start": execution_start,
            "execution_finish": execution_finish,
            "execution_time": execution_time
        })

    completion_times = np.array([
        completion[int(task_id)]
        for task_id in tasks.task_id
    ])

    deadlines = tasks.deadline.to_numpy()

    transmission_times = tx_records["tx_duration"].to_numpy()

    return {
        "Makespan": max(vm_available),
        "Average Completion Time": np.mean(completion_times),
        "Average Transmission Time": np.mean(transmission_times),
        "Energy (J)": total_energy,
        "Deadline Misses": int(
            np.sum(completion_times > deadlines)
        )
    }


# -------------------------------
# Run experiments
# -------------------------------

tx_records = calculate_transmission_times(tasks)

algorithms = {
    "FCFS": fcfs_schedule,
    "Round Robin": round_robin_schedule,
    "Min-Min": min_min_schedule
}

results = []

for name, scheduler in algorithms.items():

    assignments = scheduler(tasks, vms)

    metrics = evaluate_schedule(
        tasks,
        vms,
        assignments,
        tx_records
    )

    metrics["Algorithm"] = name
    results.append(metrics)

    print(f"\n{name} Wireless Cloud Scheduling Results")
    print("----------------------------------------")

    for key, value in metrics.items():
       
       if isinstance(value, (int, float, np.integer, np.floating)):
          print(f"{key}: {value:.4f}")
       else:
          print(f"{key}: {value}")


# -------------------------------
# Save experiment results
# -------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/wireless_cloud_results.csv",
    index=False
)


# -------------------------------
# Plot performance comparisons
# -------------------------------

metrics_to_plot = [
    ("Makespan", "Makespan (seconds)"),
    ("Average Completion Time", "Average Completion Time (seconds)"),
    ("Energy (J)", "Energy (Joules)"),
    ("Deadline Misses", "Deadline Misses")
]

for metric, ylabel in metrics_to_plot:

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Algorithm"],
        results_df[metric]
    )

    plt.xlabel("Scheduling Algorithm")
    plt.ylabel(ylabel)
    plt.title(f"Wireless Cloud Scheduling: {metric}")

    plt.grid(axis="y")
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.savefig(
        f"results/wireless_{filename}.png",
        dpi=300
    )

    plt.show()


print("\nShared wireless cloud simulation completed!")
