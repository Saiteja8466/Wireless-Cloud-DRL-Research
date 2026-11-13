
import os
import numpy as np
import pandas as pd

from wireless_cloud_ppo_env import WirelessCloudEnv


SEED = 42
NUM_TASKS = 200
NUM_VMS = 5


def generate_workload():
    """Generate the same seeded workload for every scheduler."""

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

    # Shared FIFO transmission queue.
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
    """Arrival-order tasks; choose the VM with earliest availability."""

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


def schedule_random(tasks):
    """Reproducible random VM assignment."""

    rng = np.random.default_rng(SEED)

    return [
        (i, int(rng.integers(0, NUM_VMS)))
        for i in range(len(tasks))
    ]


def schedule_online_earliest_finish(tasks, vm_mips):
    """
    Online greedy policy.

    Process tasks in arrival order and choose the VM
    with the earliest predicted completion for the
    current task.
    """

    vm_available = np.zeros(NUM_VMS)
    assignments = []

    for i, task in enumerate(tasks):
        predicted = []

        for vm in range(NUM_VMS):
            execution = task["cpu_mi"] / vm_mips[vm]

            start = max(
                task["tx_finish"],
                vm_available[vm]
            )

            predicted.append(start + execution)

        vm = int(np.argmin(predicted))

        execution = task["cpu_mi"] / vm_mips[vm]
        start = max(task["tx_finish"], vm_available[vm])

        vm_available[vm] = start + execution
        assignments.append((i, vm))

    return assignments


def schedule_min_min_offline(tasks, vm_mips):
    """
    Offline Min-Min.

    This method can inspect all unscheduled tasks,
    including future arrivals. It is an offline
    reference, not a directly equivalent online policy.
    """

    vm_available = np.zeros(NUM_VMS)
    unscheduled = set(range(len(tasks)))
    assignments = []

    while unscheduled:
        best_task = None
        best_vm = None
        best_finish = float("inf")

        for task_index in unscheduled:
            task = tasks[task_index]

            for vm in range(NUM_VMS):
                execution = task["cpu_mi"] / vm_mips[vm]

                start = max(
                    task["tx_finish"],
                    vm_available[vm]
                )

                finish = start + execution

                if finish < best_finish:
                    best_finish = finish
                    best_task = task_index
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


def validate_assignments(assignments, num_tasks):
    """Check that every task is assigned exactly once."""

    task_ids = [task_id for task_id, vm in assignments]
    vm_ids = [vm for task_id, vm in assignments]

    assert len(assignments) == num_tasks, (
        "Incorrect number of assignments"
    )

    assert len(set(task_ids)) == num_tasks, (
        "A task has been assigned more than once"
    )

    assert set(task_ids) == set(range(num_tasks)), (
        "Some tasks are missing"
    )

    assert all(0 <= vm < NUM_VMS for vm in vm_ids), (
        "Invalid VM index found"
    )

    return True


def calculate_metrics(tasks, assignments, vm_mips, vm_power):
    """Replay assignments and calculate performance metrics."""

    vm_available = np.zeros(NUM_VMS)

    completion_times = []
    response_times = []
    energy_values = []

    deadline_misses = 0

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
        "Average_Response_s": np.mean(response_times),
        "Deadline_Misses": deadline_misses,
        "Deadline_Miss_Rate_pct": (
            deadline_misses / len(tasks) * 100
        ),
        "Average_Energy_J": np.mean(energy_values),
        "Total_Energy_J": np.sum(energy_values)
    }


def main():
    print("\nSCHEDULING MODEL AUDIT")
    print("=" * 60)

    tasks, vm_mips, vm_power = generate_workload()

    print("\nCHECK 1: Workload")
    print("Task count:", len(tasks))
    print("VM count:", len(vm_mips))

    assert len(tasks) == NUM_TASKS
    assert len(vm_mips) == NUM_VMS

    print("PASS: Expected workload dimensions")

    print("\nCHECK 2: Task arrival order")

    arrivals = np.array([
        task["arrival"] for task in tasks
    ])

    arrival_sorted = np.all(np.diff(arrivals) >= 0)

    print("Chronological arrivals:", arrival_sorted)
    assert arrival_sorted

    print("PASS: Tasks are in chronological arrival order")

    print("\nCHECK 3: Shared FIFO transmission queue")

    tx_starts = np.array([
        task["tx_start"] for task in tasks
    ])

    tx_finishes = np.array([
        task["tx_finish"] for task in tasks
    ])

    tx_order_valid = np.all(np.diff(tx_finishes) >= -1e-12)

    queue_valid = np.all(
        tx_starts[1:] >= tx_finishes[:-1] - 1e-12
    )

    print("Transmission finish order:", tx_order_valid)
    print("No overlapping transmissions:", queue_valid)

    assert tx_order_valid
    assert queue_valid

    print("PASS: Shared FIFO transmission queue")

    print("\nCHECK 4: Scheduler assignments")

    schedulers = {
        "FCFS": lambda: schedule_fcfs(tasks, vm_mips),
        "Round Robin": lambda: schedule_round_robin(tasks),
        "Min-Min Offline": lambda: schedule_min_min_offline(
            tasks, vm_mips
        ),
        "Random": lambda: schedule_random(tasks),
        "Online Earliest Finish": lambda: (
            schedule_online_earliest_finish(tasks, vm_mips)
        )
    }

    results = []

    for name, scheduler in schedulers.items():
        assignments = scheduler()

        validate_assignments(assignments, len(tasks))

        metrics = calculate_metrics(
            tasks,
            assignments,
            vm_mips,
            vm_power
        )

        metrics["Scheduler"] = name
        results.append(metrics)

        print(f"PASS: {name} - all tasks assigned exactly once")

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

    os.makedirs("results", exist_ok=True)

    output_path = os.path.join(
        "results",
        "scheduling_audit_results.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nAUDIT RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\nSaved:", output_path)

    print("\nIMPORTANT:")
    print("- FCFS and Round Robin process tasks in arrival order.")
    print("- Online Earliest Finish uses only the current task.")
    print("- Min-Min Offline can inspect the complete workload.")
    print("- These are simulation results, not real cloud measurements.")

    print("\nSCHEDULING AUDIT COMPLETED")


if __name__ == "__main__":
    main()
