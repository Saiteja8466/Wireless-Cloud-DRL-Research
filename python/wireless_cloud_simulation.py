
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ==========================================
# 1. SIMULATION CONFIGURATION
# ==========================================

NUM_TASKS = 500
NUM_VMS = 5

BANDWIDTH_HZ = 20e6
TRANSMISSION_POWER_W = 0.5

# ==========================================
# 2. GENERATE CLOUD TASKS
# ==========================================

# Random task arrivals
arrival_times = np.cumsum(
    np.random.exponential(scale=1.0, size=NUM_TASKS)
)

tasks = pd.DataFrame({
    "task_id": np.arange(NUM_TASKS),
    "arrival_time": arrival_times,
    "instructions_mi": np.random.randint(
        500, 10000, NUM_TASKS
    ),
    "input_size_mb": np.random.uniform(
        0.1, 2.0, NUM_TASKS
    ),
    "deadline": np.random.uniform(
        5, 30, NUM_TASKS
    ),
    "snr_db": np.random.uniform(
        5, 25, NUM_TASKS
    )
})

# Virtual machines
vms = pd.DataFrame({
    "vm_id": np.arange(NUM_VMS),
    "mips": [1000, 1500, 2000, 2500, 3000],
    "power_w": [50, 65, 80, 95, 110]
})


# ==========================================
# 3. SHARED WIRELESS CHANNEL
# ==========================================

def calculate_transmission_schedule(tasks):

    channel_available = 0.0
    records = []

    for _, task in tasks.sort_values(
        "arrival_time"
    ).iterrows():

        snr_linear = 10 ** (task.snr_db / 10)

        data_rate = BANDWIDTH_HZ * np.log2(
            1 + snr_linear
        )

        data_bits = task.input_size_mb * 8e6

        tx_duration = data_bits / data_rate

        tx_start = max(
            task.arrival_time,
            channel_available
        )

        tx_finish = tx_start + tx_duration

        channel_available = tx_finish

        records.append({
            "task_id": int(task.task_id),
            "tx_start": tx_start,
            "tx_finish": tx_finish,
            "tx_duration": tx_duration
        })

    return pd.DataFrame(records)


# ==========================================
# 4. FCFS SCHEDULING
# ==========================================

def fcfs_schedule(ready_tasks, vms):

    vm_available = np.zeros(len(vms))
    assignments = {}

    for _, task in ready_tasks.iterrows():

        vm_index = int(np.argmin(vm_available))

        assignments[int(task.task_id)] = vm_index

        execution = (
            task.instructions_mi /
            vms.iloc[vm_index].mips
        )

        start = max(
            vm_available[vm_index],
            task.tx_finish
        )

        vm_available[vm_index] = start + execution

    return assignments


# ==========================================
# 5. ROUND ROBIN SCHEDULING
# ==========================================

def round_robin_schedule(ready_tasks, vms):

    assignments = {}

    for position, task_id in enumerate(
        ready_tasks.task_id
    ):

        assignments[int(task_id)] = (
            position % len(vms)
        )

    return assignments


# ==========================================
# 6. RELEASE-TIME-AWARE MIN-MIN
# ==========================================

def min_min_schedule(ready_tasks, vms):

    remaining = ready_tasks.copy()

    vm_available = np.zeros(len(vms))
    assignments = {}

    while not remaining.empty:

        best_task = None
        best_vm = None
        best_finish = float("inf")

        for _, task in remaining.iterrows():

            for vm_index, vm in vms.iterrows():

                execution = (
                    task.instructions_mi / vm.mips
                )

                start = max(
                    vm_available[vm_index],
                    task.tx_finish
                )

                finish = start + execution

                if finish < best_finish:

                    best_finish = finish
                    best_task = int(task.task_id)
                    best_vm = vm_index

        assignments[best_task] = best_vm

        selected_task = remaining[
            remaining.task_id == best_task
        ].iloc[0]

        execution = (
            selected_task.instructions_mi /
            vms.iloc[best_vm].mips
        )

        start = max(
            vm_available[best_vm],
            selected_task.tx_finish
        )

        vm_available[best_vm] = start + execution

        remaining = remaining[
            remaining.task_id != best_task
        ]

    return assignments


# ==========================================
# 7. PERFORMANCE EVALUATION
# ==========================================

def evaluate_schedule(
    tasks, vms, assignments, tx_records
):

    vm_available = np.zeros(len(vms))

    tx_lookup = tx_records.set_index("task_id")

    records = []

    total_energy = 0.0

    for _, task in tasks.sort_values(
        "arrival_time"
    ).iterrows():

        task_id = int(task.task_id)
        vm_index = assignments[task_id]

        vm = vms.iloc[vm_index]

        tx_start = tx_lookup.loc[
            task_id, "tx_start"
        ]

        tx_finish = tx_lookup.loc[
            task_id, "tx_finish"
        ]

        tx_duration = tx_lookup.loc[
            task_id, "tx_duration"
        ]

        execution = (
            task.instructions_mi / vm.mips
        )

        execution_start = max(
            tx_finish,
            vm_available[vm_index]
        )

        execution_finish = (
            execution_start + execution
        )

        vm_available[vm_index] = execution_finish

        end_to_end_time = (
            execution_finish - task.arrival_time
        )

        deadline_missed = (
            end_to_end_time > task.deadline
        )

        tx_energy = (
            TRANSMISSION_POWER_W * tx_duration
        )

        processing_energy = (
            vm.power_w * execution
        )

        total_energy += (
            tx_energy + processing_energy
        )

        records.append({
            "task_id": task_id,
            "arrival_time": task.arrival_time,
            "tx_start": tx_start,
            "tx_finish": tx_finish,
            "tx_wait": tx_start - task.arrival_time,
            "tx_duration": tx_duration,
            "execution_start": execution_start,
            "execution_finish": execution_finish,
            "execution_time": execution,
            "end_to_end_time": end_to_end_time,
            "deadline": task.deadline,
            "deadline_missed": deadline_missed
        })

    details = pd.DataFrame(records)

    metrics = {
        "Makespan": (
            details.execution_finish.max()
            - tasks.arrival_time.min()
        ),
        "Average Completion Time": (
            details.end_to_end_time.mean()
        ),
        "Average Transmission Queue Wait": (
            details.tx_wait.mean()
        ),
        "Average Transmission Time": (
            details.tx_duration.mean()
        ),
        "Average Execution Time": (
            details.execution_time.mean()
        ),
        "Energy (J)": total_energy,
        "Deadline Misses": int(
            details.deadline_missed.sum()
        ),
        "Deadline Miss Rate (%)": (
            details.deadline_missed.mean() * 100
        )
    }

    return metrics, details


# ==========================================
# 8. RUN ALL SCHEDULING ALGORITHMS
# ==========================================

tx_records = calculate_transmission_schedule(tasks)

ready_tasks = tasks.merge(
    tx_records[["task_id", "tx_start", "tx_finish"]],
    on="task_id"
)

ready_tasks = ready_tasks.sort_values(
    "tx_finish"
).reset_index(drop=True)

algorithms = {
    "FCFS": fcfs_schedule,
    "Round Robin": round_robin_schedule,
    "Min-Min": min_min_schedule
}

results = []
all_details = []

for name, scheduler in algorithms.items():

    assignments = scheduler(ready_tasks, vms)

    metrics, details = evaluate_schedule(
        tasks,
        vms,
        assignments,
        tx_records
    )

    metrics["Algorithm"] = name
    details["Algorithm"] = name

    results.append(metrics)
    all_details.append(details)

    print(f"\n{name} Scheduling Results")
    print("--------------------------------")

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"{key}: {value:.4f}")
        else:       
            print(f"{key}: {value}")

# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/wireless_cloud_results.csv",
    index=False
)

details_df = pd.concat(
    all_details,
    ignore_index=True
)

details_df.to_csv(
    "results/wireless_task_details.csv",
    index=False
)


# ==========================================
# 10. GENERATE COMPARISON GRAPHS
# ==========================================

plots = [
    ("Makespan", "Makespan (seconds)"),
    (
        "Average Completion Time",
        "Average End-to-End Time (seconds)"
    ),
    ("Energy (J)", "Energy (Joules)"),
    (
        "Deadline Miss Rate (%)",
        "Deadline Miss Rate (%)"
    )
]

for metric, ylabel in plots:

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Algorithm"],
        results_df[metric]
    )

    plt.xlabel("Scheduling Algorithm")
    plt.ylabel(ylabel)

    plt.title(
        f"Wireless Cloud Scheduling: {metric}"
    )

    plt.grid(axis="y")
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )

    plt.savefig(
        f"results/milestone62_{filename}.png",
        dpi=300
    )

    plt.show()


print("\nMilestone 6.2 simulation completed!")
