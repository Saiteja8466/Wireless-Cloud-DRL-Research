
import os
import numpy as np
import pandas as pd

from wireless_cloud_ppo_env import WirelessCloudEnv


SEEDS = list(range(101, 121))
NUM_TASKS = 200
NUM_VMS = 5

# Approximate two-sided 95% Student-t critical value
# for 20 independent seeds (19 degrees of freedom).
T_CRITICAL_95 = 2.093

OUTPUT_DIR = "results"


def generate_workload(seed):
    """Generate a reproducible workload for one seed."""

    env = WirelessCloudEnv(
        num_tasks=NUM_TASKS,
        num_vms=NUM_VMS
    )

    env.reset(seed=seed)

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


def schedule_fcfs(tasks, vm_mips):
    """Arrival-order tasks; assign to earliest available VM."""

    vm_available = np.zeros(NUM_VMS)
    assignments = []

    for i, task in enumerate(tasks):
        vm = int(np.argmin(vm_available))

        execution = task["cpu_mi"] / vm_mips[vm]
        start = max(task["tx_finish"], vm_available[vm])

        vm_available[vm] = start + execution
        assignments.append((i, vm))

    return assignments


def schedule_round_robin(tasks):
    """Assign tasks cyclically to VMs."""

    return [
        (i, i % NUM_VMS)
        for i in range(len(tasks))
    ]


def schedule_random(tasks, seed):
    """Random VM assignment with a reproducible seed."""

    rng = np.random.default_rng(seed + 10000)

    return [
        (i, int(rng.integers(0, NUM_VMS)))
        for i in range(len(tasks))
    ]


def schedule_online_earliest_finish(tasks, vm_mips):
    """Choose the VM with earliest predicted completion for each task."""

    vm_available = np.zeros(NUM_VMS)
    assignments = []

    for i, task in enumerate(tasks):
        predicted_finish = []

        for vm in range(NUM_VMS):
            execution = task["cpu_mi"] / vm_mips[vm]
            start = max(task["tx_finish"], vm_available[vm])

            predicted_finish.append(start + execution)

        vm = int(np.argmin(predicted_finish))

        execution = task["cpu_mi"] / vm_mips[vm]
        start = max(task["tx_finish"], vm_available[vm])

        vm_available[vm] = start + execution
        assignments.append((i, vm))

    return assignments


def schedule_min_min_offline(tasks, vm_mips):
    """Offline Min-Min with complete workload knowledge."""

    vm_available = np.zeros(NUM_VMS)
    unscheduled = set(range(len(tasks)))
    assignments = []

    while unscheduled:
        best_task = None
        best_vm = None
        best_finish = float("inf")

        for task_id in unscheduled:
            task = tasks[task_id]

            for vm in range(NUM_VMS):
                execution = task["cpu_mi"] / vm_mips[vm]
                start = max(task["tx_finish"], vm_available[vm])
                finish = start + execution

                if finish < best_finish:
                    best_finish = finish
                    best_task = task_id
                    best_vm = vm

        task = tasks[best_task]
        execution = task["cpu_mi"] / vm_mips[best_vm]

        start = max(
            task["tx_finish"],
            vm_available[best_vm]
        )

        vm_available[best_vm] = start + execution

        assignments.append((best_task, best_vm))
        unscheduled.remove(best_task)

    return assignments


def calculate_metrics(tasks, assignments, vm_mips, vm_power):
    """Calculate performance metrics by replaying VM assignments."""

    vm_available = np.zeros(NUM_VMS)

    completion_times = []
    response_times = []
    energy_values = []

    deadline_misses = 0

    task_ids = [task_id for task_id, vm in assignments]

    assert len(task_ids) == len(tasks)
    assert len(set(task_ids)) == len(tasks)
    assert set(task_ids) == set(range(len(tasks)))

    assert all(
        0 <= vm < NUM_VMS
        for task_id, vm in assignments
    )

    for task_id, vm in assignments:
        task = tasks[task_id]

        execution = task["cpu_mi"] / vm_mips[vm]

        start = max(
            task["tx_finish"],
            vm_available[vm]
        )

        completion = start + execution
        vm_available[vm] = completion

        response = completion - task["arrival"]

        execution_energy = vm_power[vm] * execution
        total_energy = task["tx_energy"] + execution_energy

        deadline_misses += int(response > task["deadline"])

        completion_times.append(completion)
        response_times.append(response)
        energy_values.append(total_energy)

    return {
        "Makespan_s": max(completion_times),
        "Average_Response_s": float(np.mean(response_times)),
        "Deadline_Misses": int(deadline_misses),
        "Deadline_Miss_Rate_pct": (
            deadline_misses / len(tasks) * 100.0
        ),
        "Average_Energy_J": float(np.mean(energy_values)),
        "Total_Energy_J": float(np.sum(energy_values))
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_results = []

    print("\nMULTI-SEED SCHEDULER VALIDATION")
    print("=" * 65)
    print("Independent workload seeds:", len(SEEDS))
    print("Tasks per workload:", NUM_TASKS)
    print("VMs:", NUM_VMS)

    for run_number, seed in enumerate(SEEDS, start=1):
        tasks, vm_mips, vm_power = generate_workload(seed)

        # Verify chronological task arrivals.
        arrivals = np.array([
            task["arrival"] for task in tasks
        ])

        assert np.all(np.diff(arrivals) >= 0)

        # Verify shared FIFO transmission queue.
        tx_starts = np.array([
            task["tx_start"] for task in tasks
        ])

        tx_finishes = np.array([
            task["tx_finish"] for task in tasks
        ])

        assert np.all(np.diff(tx_finishes) >= -1e-12)
        assert np.all(
            tx_starts[1:] >= tx_finishes[:-1] - 1e-12
        )

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

        for name, scheduler in schedulers.items():
            assignments = scheduler()

            metrics = calculate_metrics(
                tasks,
                assignments,
                vm_mips,
                vm_power
            )

            raw_results.append({
                "Seed": seed,
                "Scheduler": name,
                **metrics
            })

        print(
            f"Completed seed {seed} "
            f"({run_number}/{len(SEEDS)})"
        )

    raw_df = pd.DataFrame(raw_results)

    raw_path = os.path.join(
        OUTPUT_DIR,
        "multi_seed_raw_results.csv"
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

    summary_rows = []

    for scheduler in raw_df["Scheduler"].unique():
        subset = raw_df[
            raw_df["Scheduler"] == scheduler
        ]

        row = {
            "Scheduler": scheduler,
            "Number_of_Seeds": len(subset)
        }

        for metric in metric_columns:
            values = subset[metric].to_numpy(dtype=float)

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1))

            standard_error = std_value / np.sqrt(len(values))

            margin = T_CRITICAL_95 * standard_error

            row[f"{metric}_Mean"] = mean_value
            row[f"{metric}_Std"] = std_value
            row[f"{metric}_CI95_Lower"] = mean_value - margin
            row[f"{metric}_CI95_Upper"] = mean_value + margin

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "multi_seed_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # Concise console summary.
    display_columns = [
        "Scheduler",
        "Makespan_s_Mean",
        "Makespan_s_Std",
        "Average_Response_s_Mean",
        "Average_Response_s_Std",
        "Deadline_Miss_Rate_pct_Mean",
        "Average_Energy_J_Mean"
    ]

    print("\nMULTI-SEED SUMMARY")
    print("=" * 100)

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n95% CONFIDENCE INTERVALS")
    print("=" * 100)

    for _, row in summary_df.iterrows():
        print(f"\n{row['Scheduler']}")

        for metric in [
            "Makespan_s",
            "Average_Response_s",
            "Deadline_Miss_Rate_pct",
            "Average_Energy_J"
        ]:
            print(
                f"{metric}: "
                f"{row[f'{metric}_Mean']:.4f} "
                f"[{row[f'{metric}_CI95_Lower']:.4f}, "
                f"{row[f'{metric}_CI95_Upper']:.4f}]"
            )

    print("\nFILES SAVED:")
    print(raw_path)
    print(summary_path)

    print("\nMULTI-SEED VALIDATION COMPLETED")

    print(
        "\nNOTE: Min-Min is an offline benchmark. "
        "The other policies use arrival-order task decisions."
    )

    print(
        "Results are simulation estimates, not real cloud measurements."
    )


if __name__ == "__main__":
    main()
