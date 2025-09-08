# envs/cloud_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.workloads import generate_workload

class CloudCostGym(gym.Env):
    """
    Gym environment wrapper for cloud cost optimization.
    - Observations: demand, instances, utilization, latency
    - Actions: scale down (-1), hold (0), scale up (+1)
    - Reward: -(infra_cost + SLA penalty)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_steps=1440, seed=None):
        super().__init__()
        self.n_steps = n_steps
        self.capacity_per_instance = 150
        self.price_per_instance = 0.9
        self.latency_target = 200
        self.sla_penalty = 2.0
        self.boot_lag = 3
        self.rng = np.random.default_rng(seed)

        # Gym spaces
        self.action_space = spaces.Discrete(3)  # {-1, 0, +1}
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.reset(seed=seed)

    def _get_obs(self):
        idx = min(self.t, len(self.workload) - 1)
        demand = self.workload[idx]
        capacity = self.instances * self.capacity_per_instance
        utilization = min(1.0, demand / capacity) if capacity > 0 else 1.0
        return np.array([demand, self.instances, utilization, self.latency], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.instances = 3
        self.pending = []
        self.workload = generate_workload(self.n_steps, seed=seed)
        self.latency = 100.0
        self.history = {"demand": [], "instances": [], "latency": [], "cost": []}
        return self._get_obs(), {}

    def step(self, action):
        # If episode already done, return terminal observation without advancing
        if self.t >= self.n_steps:
            return self._get_obs(), 0.0, True, False, {}
        # Handle scaling
        if action == 0:   # scale down
            self.instances = max(1, self.instances - 1)
        elif action == 2: # scale up
            self.pending.append(self.boot_lag)

        # Activate pending instances
        self.pending = [p - 1 for p in self.pending]
        ready = [p for p in self.pending if p <= 0]
        self.instances += len(ready)
        self.pending = [p for p in self.pending if p > 0]

        # Current demand
        # Guard against out-of-range access
        idx = min(self.t, len(self.workload) - 1)
        d = self.workload[idx]
        cap = self.instances * self.capacity_per_instance
        util = d / cap if cap > 0 else 1.0

        # Latency model
        if util <= 0.6:
            latency = 120
        elif util <= 0.8:
            latency = 120 + (util - 0.6) * 300
        else:
            latency = 180 + (util - 0.8) * 1000
        self.latency = latency

        # Costs
        infra_cost = self.instances * self.price_per_instance
        penalty = self.sla_penalty if latency > self.latency_target else 0
        reward = - (infra_cost + penalty)

        # Save history
        self.history["demand"].append(d)
        self.history["instances"].append(self.instances)
        self.history["latency"].append(latency)
        self.history["cost"].append(infra_cost)

        # Advance time
        self.t += 1
        done = self.t >= self.n_steps
        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Step {self.t}, Instances={self.instances}, Latency={self.latency:.1f}ms")
