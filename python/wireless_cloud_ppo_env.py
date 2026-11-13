
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WirelessCloudEnv(gym.Env):
    """
    Wireless-cloud task scheduling environment.

    Wireless transmission is represented using a Shannon-capacity
    abstraction. Cloud execution is represented using VM MIPS and
    power models. This is a simulation, not real radio measurement.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_tasks=200, num_vms=5):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_vms = num_vms

        self.bandwidth_hz = 20e6
        self.noise_power_w = 1e-10
        self.radio_power_w = 0.5

        self.vm_mips = np.array(
            [1000, 1500, 2000, 2500, 3000],
            dtype=np.float32
        )[:num_vms]

        self.vm_power = np.array(
            [50, 65, 80, 95, 110],
            dtype=np.float32
        )[:num_vms]

        self.action_space = spaces.Discrete(num_vms)

        # Observation:
        # task size, CPU demand, deadline, SNR, arrival gap,
        # wireless queue availability, and VM availability,
        # VM processing capacities and power.
        obs_size = 5 + 1 + num_vms * 3

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.current_task = 0
        self.vm_available = np.zeros(num_vms, dtype=np.float64)
        self.tx_available = 0.0
        self.previous_arrival = 0.0
        self.tasks = []

    def _generate_tasks(self):
        tasks = []
        arrival = 0.0

        for _ in range(self.num_tasks):
            arrival += self.np_random.exponential(0.5)

            tasks.append({
                "arrival": arrival,
                "size_mb": self.np_random.uniform(0.1, 2.0),
                "cpu_mi": self.np_random.uniform(500, 5000),
                "deadline": self.np_random.uniform(3.0, 12.0),
                "snr_db": self.np_random.uniform(0.0, 25.0)
            })

        return tasks

    def _get_obs(self):
        task = self.tasks[self.current_task]

        arrival = task["arrival"]
        arrival_gap = max(0.0, arrival - self.previous_arrival)

        vm_wait = np.maximum(
            0.0,
            self.vm_available - arrival
        )

        tx_wait = max(0.0, self.tx_available - arrival)

        obs = [
            task["size_mb"] / 2.0,
            task["cpu_mi"] / 5000.0,
            task["deadline"] / 12.0,
            task["snr_db"] / 25.0,
            min(arrival_gap / 2.0, 1.0),
            min(tx_wait / 10.0, 1.0)
        ]

        for i in range(self.num_vms):
            obs.extend([
                min(vm_wait[i] / 10.0, 1.0),
                self.vm_mips[i] / 3000.0,
                self.vm_power[i] / 110.0
            ])

        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_task = 0
        self.vm_available = np.zeros(
            self.num_vms, dtype=np.float64
        )
        self.tx_available = 0.0
        self.previous_arrival = 0.0

        self.tasks = self._generate_tasks()

        return self._get_obs(), {}

    def step(self, action):
        task = self.tasks[self.current_task]
        vm = int(action)

        arrival = task["arrival"]

        # Wireless transmission model
        snr_linear = 10 ** (task["snr_db"] / 10.0)

        rate_bps = self.bandwidth_hz * np.log2(
            1.0 + snr_linear
        )

        data_bits = task["size_mb"] * 8e6
        tx_start = max(arrival, self.tx_available)
        tx_time = data_bits / rate_bps
        tx_finish = tx_start + tx_time

        self.tx_available = tx_finish

        # Cloud VM execution model
        execution_time = task["cpu_mi"] / self.vm_mips[vm]

        execution_start = max(
            tx_finish,
            self.vm_available[vm]
        )

        completion_time = execution_start + execution_time
        self.vm_available[vm] = completion_time

        response_time = completion_time - arrival

        deadline_missed = (
            response_time > task["deadline"]
        )

        # Energy model
        transmission_energy = (
            self.radio_power_w * tx_time
        )

        execution_energy = (
            self.vm_power[vm] * execution_time
        )

        total_energy = (
            transmission_energy + execution_energy
        )

        # Reward: lower response time, deadline misses and energy
        reward = -(
            0.5 * min(response_time / 12.0, 3.0)
            + 2.0 * float(deadline_missed)
            + 0.05 * min(total_energy / 500.0, 3.0)
        )

        self.previous_arrival = arrival
        self.current_task += 1

        terminated = (
            self.current_task >= self.num_tasks
        )
        truncated = False

        info = {
            "vm_selected": vm,
            "snr_db": task["snr_db"],
            "tx_time": tx_time,
            "execution_time": execution_time,
            "response_time": response_time,
            "deadline": task["deadline"],
            "deadline_missed": deadline_missed,
            "energy_j": total_energy,
            "completion_time": completion_time
        }

        if terminated:
            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32
            )
        else:
            observation = self._get_obs()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )
