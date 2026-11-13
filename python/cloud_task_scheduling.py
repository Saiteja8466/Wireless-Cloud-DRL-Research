
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

NUM_TASKS = 500
NUM_VMS = 5

# Generate cloud tasks
tasks = pd.DataFrame({
    "task_id": np.arange(NUM_TASKS),
    "instructions_mi": np.random.randint(500, 10000, NUM_TASKS),
    "input_size_mb": np.random.randint(10, 500, NUM_TASKS),
    "deadline": np.random.randint(5, 50, NUM_TASKS)
})

# Virtual machines
vms = pd.DataFrame({
    "vm_id": np.arange(NUM_VMS),
    "mips": [1000, 1500, 2000, 2500, 3000],
    "power_w": [50, 65, 80, 95, 110]
})


# FCFS: earliest available VM
def fcfs_schedule(tasks, vms):
    available = np.zeros(len(vms))
    assignments = {}

    for _, task in tasks.iterrows():
        vm = int(np.argmin(available))
        assignments[int(task.task_id)] = vm

        available[vm] += (
            task.instructions_mi / vms.iloc[vm].mips
        )

    return assignments


# Round Robin
def round_robin_schedule(tasks, vms):
    assignments = {}

    for position, task_id in enumerate(tasks.task_id):
        assignments[int(task_id)] = position % len(vms)

    return assignments


# Min-Min scheduling
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


# Evaluate assignments
def evaluate_schedule(tasks, vms, assignments):

    available = np.zeros(len(vms))
    completion = {}
    energy = 0.0

    for _, task in tasks.iterrows():
        task_id = int(task.task_id)
        vm_index = assignments[task_id]
        vm = vms.iloc[vm_index]

        execution = task.instructions_mi / vm.mips
        finish = available[vm_index] + execution

        available[vm_index] = finish
        completion[task_id] = finish

        energy += vm.power_w * execution

    completion_times = np.array([
        completion[int(task_id)]
        for task_id in tasks.task_id
    ])

    deadlines = tasks.deadline.to_numpy()

    return {
        "Makespan": max(available),
        "Average Completion Time": np.mean(completion_times),
        "Energy (J)": energy,
        "Deadline Misses": int(
            np.sum(completion_times > deadlines)
        )
    }


# Run algorithms
algorithms = {
    "FCFS": fcfs_schedule,
    "Round Robin": round_robin_schedule,
    "Min-Min": min_min_schedule
}

results = []

for name, scheduler in algorithms.items():

    assignments = scheduler(tasks, vms)

    metrics = evaluate_schedule(
        tasks, vms, assignments
    )

    metrics["Algorithm"] = name
    results.append(metrics)

    print(f"\n{name} Scheduling Results")
    print("-----------------------------")

    for key, value in metrics.items():
        if key != "Algorithm":
            print(f"{key}: {value:.4f}")


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/cloud_scheduling_results.csv",
    index=False
)


# Makespan graph
plt.figure(figsize=(8, 5))
plt.bar(results_df["Algorithm"], results_df["Makespan"])
plt.xlabel("Scheduling Algorithm")
plt.ylabel("Makespan (seconds)")
plt.title("Cloud Scheduling Makespan Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/cloud_makespan_comparison.png", dpi=300)
plt.show()


# Energy graph
plt.figure(figsize=(8, 5))
plt.bar(results_df["Algorithm"], results_df["Energy (J)"])
plt.xlabel("Scheduling Algorithm")
plt.ylabel("Energy (Joules)")
plt.title("Cloud Scheduling Energy Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/cloud_energy_comparison.png", dpi=300)
plt.show()

print("\nCorrected cloud scheduling experiment completed!")
