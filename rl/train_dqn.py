# rl/train_dqn.py
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from envs.cloud_gym import CloudCostGym

def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_results(env, title="RL Agent Evaluation", save_path: str | None = None):
    """Plot demand, capacity, latency, and cumulative cost. If save_path is provided, save PNG there."""
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
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = ensure_output_dir(os.path.join(os.path.dirname(__file__), "..", "outputs"))

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
    model_path = os.path.join(out_dir, "dqn_cloud_cost")
    model.save(model_path)

    # Evaluate trained model
    print("Evaluating RL agent...")
    eval_env = CloudCostGym(n_steps=300, seed=123)
    obs, _ = eval_env.reset()
    total_reward, total_cost, violations = 0.0, 0.0, 0

    for _ in range(eval_env.n_steps):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = eval_env.step(int(action))
        total_reward += float(reward)
        total_cost += float(eval_env.history["cost"][-1])
        if eval_env.history["latency"][-1] > eval_env.latency_target:
            violations += 1
        if done:
            break

    print(f"Total Reward = {total_reward:.2f}")
    print(f"Total Cost   = {total_cost:.2f}")
    print(f"SLA Violations = {violations}/{eval_env.n_steps}")

    # Save metrics and plots
    metrics = {
        "total_reward": total_reward,
        "total_cost": total_cost,
        "sla_violations": violations,
        "steps": eval_env.t,
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save history csv
    csv_path = os.path.join(out_dir, "eval_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("minute,demand,instances,latency_ms,cost\n")
        for i in range(len(eval_env.history["demand"])):
            f.write(
                f"{i},{eval_env.history['demand'][i]},{eval_env.history['instances'][i]},"
                f"{eval_env.history['latency'][i]},{eval_env.history['cost'][i]}\n"
            )

    plot_results(eval_env, title="RL Agent Performance", save_path=os.path.join(out_dir, "rl_performance.png"))
