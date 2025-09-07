# envs/cloud_env.py
import matplotlib.pyplot as plt
import numpy as np
from envs.workloads import generate_workload

class CloudEnvironment:
    """
    A simple environment that simulates:
    - incoming workload (requests/sec)
    - servers (instances) handling requests
    - latency (p95) depending on utilization
    - cost of running instances
    """

    def __init__(self, n_steps=1440, seed=None):
        # Simulation settings
        self.n_steps = n_steps
        self.capacity_per_instance = 150   # req/s each instance can handle
        self.price_per_instance = 0.9      # cost per instance per minute
        self.latency_target = 200          # ms SLA target
        self.sla_penalty = 2.0             # penalty when SLA violated
        self.boot_lag = 3                  # minutes delay for new instance to start

        # Workload
        self.workload = generate_workload(n_steps, seed)
        self.t = 0
        self.instances = 3
        self.pending = []   # booting servers not yet active

        # Metrics
        self.history = {"demand": [], "instances": [], "latency": [], "cost": []}

    def step(self, action):
        """
        Take one step in simulation.
        action: -1 (scale down), 0 (hold), +1 (scale up)
        """
        # Handle scaling
        if action == -1:
            self.instances = max(1, self.instances - 1)
        elif action == 1:
            self.pending.append(self.boot_lag)

        # Activate pending instances
        self.pending = [p - 1 for p in self.pending]
        ready = [p for p in self.pending if p <= 0]
        self.instances += len(ready)
        self.pending = [p for p in self.pending if p > 0]

        # Current demand
        d = self.workload[self.t]
        capacity = self.instances * self.capacity_per_instance
        utilization = d / capacity if capacity > 0 else 1.0

        # Latency model
        if utilization <= 0.6:
            latency = 120
        elif utilization <= 0.8:
            latency = 120 + (utilization - 0.6) * 300
        else:
            latency = 180 + (utilization - 0.8) * 1000

        # Cost
        infra_cost = self.instances * self.price_per_instance
        penalty = self.sla_penalty if latency > self.latency_target else 0
        reward = - (infra_cost + penalty)

        # Save history
        self.history["demand"].append(d)
        self.history["instances"].append(self.instances)
        self.history["latency"].append(latency)
        self.history["cost"].append(infra_cost)

        self.t += 1
        done = self.t >= self.n_steps
        return reward, done

    def reset(self, seed=None):
        self.t = 0
        self.instances = 3
        self.pending = []
        self.workload = generate_workload(self.n_steps, seed)
        self.history = {"demand": [], "instances": [], "latency": [], "cost": []}

    def plot_history(self):
        """Visualize demand, instances, latency, and cost."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(self.history["demand"], label="Demand (req/s)")
        axes[0].plot(np.array(self.history["instances"]) * self.capacity_per_instance, 
                     label="Capacity (req/s)")
        axes[0].legend()
        axes[0].set_ylabel("Requests/sec")

        axes[1].plot(self.history["latency"], color="orange", label="Latency (ms)")
        axes[1].axhline(self.latency_target, color="red", linestyle="--", label="SLA Target")
        axes[1].legend()
        axes[1].set_ylabel("Latency (ms)")

        axes[2].plot(np.cumsum(self.history["cost"]), color="green", label="Cumulative Cost")
        axes[2].legend()
        axes[2].set_ylabel("Cost (₹)")
        axes[2].set_xlabel("Time (minutes)")

        plt.tight_layout()
        plt.show()


# Quick test
if __name__ == "__main__":
    env = CloudEnvironment(n_steps=300, seed=42)
    env.reset(seed=42)

    # Example policy: scale up if demand > capacity, scale down if demand < half capacity
    for t in range(env.n_steps):
        d = env.workload[t]
        cap = env.instances * env.capacity_per_instance
        if d > cap:
            action = 1
        elif d < cap * 0.5 and env.instances > 1:
            action = -1
        else:
            action = 0
        env.step(action)

    env.plot_history()
