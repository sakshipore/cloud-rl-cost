# rl/enhanced_train_dqn.py
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent

def ensure_output_dir(path: str) -> str:
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)
    return path

def plot_enhanced_results(env, title="Enhanced RL Agent Evaluation", save_path: str = None):
    """Plot comprehensive results including service usage and costs."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Demand vs Capacity
    axes[0].plot(env.history["demand"], label="Demand (req/s)", color="blue")
    total_capacity = []
    for i in range(len(env.history["demand"])):
        cap = 0
        for service_name in env.services.keys():
            instances = env.history["service_usage"][service_name][i]
            cap += instances * env.services[service_name].capacity
        total_capacity.append(cap)
    axes[0].plot(total_capacity, label="Total Capacity (req/s)", color="red")
    axes[0].legend()
    axes[0].set_ylabel("Requests/sec")
    axes[0].set_title("Demand vs Capacity")
    
    # Service Usage
    for service_name in env.services.keys():
        axes[1].plot(env.history["service_usage"][service_name], 
                    label=f"{service_name} instances", alpha=0.7)
    axes[1].legend()
    axes[1].set_ylabel("Instances")
    axes[1].set_title("Service Usage Over Time")
    
    # Latency
    axes[2].plot(env.history["latency"], color="orange", label="Latency (ms)")
    axes[2].axhline(env.latency_target, color="red", linestyle="--", label="SLA Target")
    axes[2].legend()
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Latency Performance")
    
    # Cumulative Cost
    cumulative_cost = [sum(env.history["total_cost"][:i+1]) for i in range(len(env.history["total_cost"]))]
    axes[3].plot(cumulative_cost, color="green", label="Cumulative Cost")
    axes[3].legend()
    axes[3].set_ylabel("Cost ($)")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_title("Cumulative Cost")
    
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_service_costs(env, save_path: str = None):
    """Plot individual service costs over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for service_name in env.services.keys():
        cumulative_service_cost = [
            sum(env.history["service_costs"][service_name][:i+1]) 
            for i in range(len(env.history["service_costs"][service_name]))
        ]
        ax.plot(cumulative_service_cost, label=f"{service_name} cost", alpha=0.8)
    
    ax.legend()
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title("Service-Specific Costs Over Time")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)

def train_enhanced_dqn(workload_type: str = "diurnal", n_steps: int = 500, 
                      total_timesteps: int = 20000, seed: int = 42):
    """
    Train DQN agent on enhanced cloud environment.
    
    Args:
        workload_type: Type of workload pattern
        n_steps: Number of steps per episode
        total_timesteps: Total training timesteps
        seed: Random seed
        
    Returns:
        Trained model and evaluation results
    """
    out_dir = ensure_output_dir(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    
    # Create training environment
    train_env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed, workload_type=workload_type)
    
    # Create evaluation environment
    eval_env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed+1, workload_type=workload_type)
    
    # Create DQN model with enhanced configuration
    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log=os.path.join(out_dir, "tensorboard_logs")
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best_model"),
        log_path=os.path.join(out_dir, "eval_logs"),
        eval_freq=2000,
        deterministic=True,
        render=False
    )
    
    print(f"Training DQN agent on {workload_type} workload...")
    print(f"Training for {total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model_path = os.path.join(out_dir, f"dqn_enhanced_{workload_type}")
    model.save(model_path)
    
    print(f"Training completed. Model saved to {model_path}")
    
    return model, eval_env

def evaluate_enhanced_model(model, env, n_episodes: int = 5):
    """
    Evaluate the trained model on multiple episodes.
    
    Args:
        model: Trained DQN model
        env: Environment for evaluation
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "episodes": [],
        "total_rewards": [],
        "total_costs": [],
        "sla_violations": [],
        "interruptions": [],
        "service_usage": {name: [] for name in env.services.keys()},
        "service_costs": {name: [] for name in env.services.keys()}
    }
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=42 + episode)
        done = False
        episode_reward = 0.0
        episode_cost = 0.0
        episode_violations = 0
        episode_interruptions = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_cost += info.get("service_costs", {}).get("total", 0)
            episode_violations += info.get("sla_violation", 0)
            episode_interruptions += info.get("interrupted", 0)
        
        # Store episode results
        results["episodes"].append(episode)
        results["total_rewards"].append(episode_reward)
        results["total_costs"].append(episode_cost)
        results["sla_violations"].append(episode_violations)
        results["interruptions"].append(episode_interruptions)
        
        # Store service usage and costs
        for service_name in env.services.keys():
            results["service_usage"][service_name].append(
                np.mean(env.history["service_usage"][service_name])
            )
            results["service_costs"][service_name].append(
                sum(env.history["service_costs"][service_name])
            )
    
    return results

def compare_with_baselines(model, env, baselines: list = None):
    """
    Compare RL model with rule-based baselines.
    
    Args:
        model: Trained DQN model
        env: Environment for evaluation
        baselines: List of baseline agent types to compare
        
    Returns:
        Dictionary with comparison results
    """
    if baselines is None:
        baselines = ["cost_optimized", "reliability_optimized", "hybrid", "workload_aware"]
    
    comparison_results = {}
    
    # Evaluate RL model
    print("Evaluating RL model...")
    rl_results = evaluate_enhanced_model(model, env, n_episodes=3)
    comparison_results["rl"] = {
        "avg_reward": np.mean(rl_results["total_rewards"]),
        "avg_cost": np.mean(rl_results["total_costs"]),
        "avg_violations": np.mean(rl_results["sla_violations"]),
        "avg_interruptions": np.mean(rl_results["interruptions"])
    }
    
    # Evaluate baselines
    for baseline_type in baselines:
        print(f"Evaluating {baseline_type} baseline...")
        baseline_agent = create_agent(baseline_type)
        
        baseline_results = {
            "total_rewards": [],
            "total_costs": [],
            "sla_violations": [],
            "interruptions": []
        }
        
        for episode in range(3):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_violations = 0
            episode_interruptions = 0
            
            while not done:
                action = baseline_agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_cost += info.get("service_costs", {}).get("total", 0)
                episode_violations += info.get("sla_violation", 0)
                episode_interruptions += info.get("interrupted", 0)
            
            baseline_results["total_rewards"].append(episode_reward)
            baseline_results["total_costs"].append(episode_cost)
            baseline_results["sla_violations"].append(episode_violations)
            baseline_results["interruptions"].append(episode_interruptions)
        
        comparison_results[baseline_type] = {
            "avg_reward": np.mean(baseline_results["total_rewards"]),
            "avg_cost": np.mean(baseline_results["total_costs"]),
            "avg_violations": np.mean(baseline_results["sla_violations"]),
            "avg_interruptions": np.mean(baseline_results["interruptions"])
        }
    
    return comparison_results

if __name__ == "__main__":
    # Train and evaluate on different workload types
    workload_types = ["diurnal", "steady", "batch", "bursty"]
    
    for workload_type in workload_types:
        print(f"\n{'='*60}")
        print(f"Training and evaluating on {workload_type} workload")
        print(f"{'='*60}")
        
        # Train model
        model, eval_env = train_enhanced_dqn(
            workload_type=workload_type,
            n_steps=300,
            total_timesteps=10000,  # Reduced for demo
            seed=42
        )
        
        # Evaluate model
        print(f"\nEvaluating {workload_type} workload...")
        results = evaluate_enhanced_model(model, eval_env, n_episodes=3)
        
        print(f"Average reward: {np.mean(results['total_rewards']):.2f}")
        print(f"Average cost: {np.mean(results['total_costs']):.2f}")
        print(f"Average SLA violations: {np.mean(results['sla_violations']):.1f}")
        print(f"Average interruptions: {np.mean(results['interruptions']):.1f}")
        
        # Compare with baselines
        print(f"\nComparing with baselines on {workload_type} workload...")
        comparison = compare_with_baselines(model, eval_env)
        
        print("\nComparison Results:")
        for strategy, metrics in comparison.items():
            print(f"{strategy}:")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Avg Cost: {metrics['avg_cost']:.2f}")
            print(f"  Avg Violations: {metrics['avg_violations']:.1f}")
            print(f"  Avg Interruptions: {metrics['avg_interruptions']:.1f}")
        
        # Save results
        out_dir = ensure_output_dir(os.path.join(os.path.dirname(__file__), "..", "outputs"))
        
        # Save evaluation results
        with open(os.path.join(out_dir, f"eval_results_{workload_type}.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Save comparison results
        with open(os.path.join(out_dir, f"comparison_results_{workload_type}.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Generate plots
        plot_enhanced_results(eval_env, 
                            title=f"Enhanced RL Agent - {workload_type.title()} Workload",
                            save_path=os.path.join(out_dir, f"rl_performance_{workload_type}.png"))
        
        plot_service_costs(eval_env, 
                          save_path=os.path.join(out_dir, f"service_costs_{workload_type}.png"))
        
        print(f"Results saved for {workload_type} workload")
    
    print(f"\n{'='*60}")
    print("Training and evaluation completed!")
    print(f"{'='*60}")
