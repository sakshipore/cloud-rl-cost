# rl/train_dqn.py
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from envs.cloud_gym import CloudCostGym

def plot_results(env, title="RL Agent Evaluation"):
    """Plot demand, capacity, latency, and cumulative cost."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Demand vs Capacity
    axes[0].plot(env.history["demand"], label="Demand (req/s)")
    capacity = [x * env.capacity_per_instance for x in env.history["instances"]]
    axes[0].plot(capacity, label="Capacity (req/s)")
    axes[0].legend()
    axes[0].set_ylabel("Requests/sec")

    # Latency
    axes[1].plot(env.history["latency"], color="orange", label="Latency (ms)")
    axes[1].axhline(env.latency_target, color="red", linestyle="--", label="SLA Target")
    axes[1].legend()
    axes[1].set_ylabel("Latency (ms)")

    # Cumulative cost
    cumulative_cost = [sum(env.history["cost"][:i+1]) for i in range(len(env.history["cost"]))]
    axes[2].plot(cumulative_cost, color="green", label="Cumulative Cost")
    axes[2].legend()
    axes[2].set_ylabel("Cost (₹)")
    axes[2].set_xlabel("Time (minutes)")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create training environment
    env = CloudCostGym(n_steps=500, seed=42)  # shorter run for demo

    # Train DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500
    )

    print("Training RL agent... (this may take a few minutes)")
    model.learn(total_timesteps=20000)  # train for 20k steps
    model.save("dqn_cloud_cost")

    # Evaluate trained model
    print("Evaluating RL agent...")
    eval_env = CloudCostGym(n_steps=300, seed=123)
    obs, _ = eval_env.reset()
    total_reward, total_cost, violations = 0, 0, 0

    for _ in range(eval_env.n_steps):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = eval_env.step(action)
        total_reward += reward
        total_cost += eval_env.history["cost"][-1]
        if eval_env.history["latency"][-1] > eval_env.latency_target:
            violations += 1
        if done:
            break

    print(f"Total Reward = {total_reward:.2f}")
    print(f"Total Cost   = {total_cost:.2f}")
    print(f"SLA Violations = {violations}/{eval_env.n_steps}")

    # Plot results
    plot_results(eval_env, title="RL Agent Performance")
