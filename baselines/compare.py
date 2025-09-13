# baselines/compare.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent, RuleBasedAgent
from stable_baselines3 import DQN

def compare_strategies(strategies: Dict[str, Any], 
                      workload_types: List[str] = None,
                      n_episodes: int = 10,
                      n_steps: int = 300,
                      output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Compare multiple strategies across different workload types.
    
    Args:
        strategies: Dictionary mapping strategy names to strategy objects
        workload_types: List of workload types to test
        n_episodes: Number of episodes per strategy per workload
        n_steps: Number of steps per episode
        output_dir: Directory to save results
        
    Returns:
        Dictionary with comparison results
    """
    if workload_types is None:
        workload_types = ["diurnal", "steady", "batch", "bursty"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_results = {}
    
    for workload_type in workload_types:
        print(f"\nEvaluating strategies on {workload_type} workload...")
        
        # Create environment for this workload type
        env = EnhancedCloudCostGym(n_steps=n_steps, seed=42, workload_type=workload_type)
        
        workload_results = {}
        
        for strategy_name, strategy in strategies.items():
            print(f"  Evaluating {strategy_name}...")
            
            results = {
                "strategy_name": strategy_name,
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
                    # Get action from strategy
                    if isinstance(strategy, RuleBasedAgent):
                        action = strategy.predict(obs)
                    else:  # RL model
                        action, _ = strategy.predict(obs, deterministic=True)
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_cost += -reward  # Reward is negative cost
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
            
            # Calculate summary statistics
            results["summary"] = {
                "avg_reward": np.mean(results["total_rewards"]),
                "std_reward": np.std(results["total_rewards"]),
                "avg_cost": np.mean(results["total_costs"]),
                "std_cost": np.std(results["total_costs"]),
                "avg_violations": np.mean(results["sla_violations"]),
                "std_violations": np.std(results["sla_violations"]),
                "avg_interruptions": np.mean(results["interruptions"]),
                "std_interruptions": np.std(results["interruptions"]),
                "sla_violation_rate": np.mean(results["sla_violations"]) / n_steps,
                "cost_per_step": np.mean(results["total_costs"]) / n_steps
            }
            
            workload_results[strategy_name] = results
        
        comparison_results[workload_type] = workload_results
    
    return comparison_results

def generate_comparison_report(comparison_results: Dict[str, Any], 
                              save_path: str = None) -> str:
    """
    Generate a comprehensive comparison report.
    
    Args:
        comparison_results: Results from compare_strategies
        save_path: Path to save the report
        
    Returns:
        Report text
    """
    report = []
    report.append("=" * 80)
    report.append("CLOUD COST OPTIMIZATION - STRATEGY COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    for workload_type, workload_results in comparison_results.items():
        report.append(f"WORKLOAD TYPE: {workload_type.upper()}")
        report.append("-" * 40)
        
        # Create comparison table
        strategies = list(workload_results.keys())
        
        # Header
        header = f"{'Strategy':<25} {'Avg Cost':<12} {'SLA Rate':<12} {'Interruptions':<15} {'Cost/Step':<12}"
        report.append(header)
        report.append("-" * len(header))
        
        # Data rows
        for strategy_name in strategies:
            results = workload_results[strategy_name]
            summary = results["summary"]
            
            row = f"{strategy_name:<25}"
            row += f"${summary['avg_cost']:.2f}".ljust(12)
            row += f"{summary['sla_violation_rate']:.1%}".ljust(12)
            row += f"{summary['avg_interruptions']:.1f}".ljust(15)
            row += f"${summary['cost_per_step']:.3f}".ljust(12)
            
            report.append(row)
        
        report.append("")
    
    # Overall summary
    report.append("OVERALL SUMMARY")
    report.append("-" * 40)
    
    # Calculate overall best performers
    overall_metrics = {
        "avg_cost": {},
        "sla_violation_rate": {},
        "avg_interruptions": {}
    }
    
    for workload_type, workload_results in comparison_results.items():
        for strategy_name, results in workload_results.items():
            summary = results["summary"]
            
            for metric in overall_metrics.keys():
                if strategy_name not in overall_metrics[metric]:
                    overall_metrics[metric][strategy_name] = []
                overall_metrics[metric][strategy_name].append(summary[metric])
    
    for metric in overall_metrics.keys():
        if overall_metrics[metric]:
            # Find best strategy (lowest for cost and interruptions, lowest for SLA rate)
            best_strategy = min(overall_metrics[metric].keys(), 
                              key=lambda x: np.mean(overall_metrics[metric][x]))
            
            report.append(f"Best {metric.replace('_', ' ').title()}: {best_strategy}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)
    
    return report_text

def plot_comparison_results(comparison_results: Dict[str, Any], 
                           save_dir: str = "outputs"):
    """
    Generate comparison plots.
    
    Args:
        comparison_results: Results from compare_strategies
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Cost comparison across workload types
    _plot_cost_comparison(comparison_results, save_dir)
    
    # Plot 2: SLA violation rate comparison
    _plot_sla_comparison(comparison_results, save_dir)
    
    # Plot 3: Service usage patterns
    _plot_service_usage(comparison_results, save_dir)
    
    # Plot 4: Performance trade-offs
    _plot_performance_tradeoffs(comparison_results, save_dir)

def _plot_cost_comparison(comparison_results: Dict[str, Any], save_dir: str):
    """Plot cost comparison across strategies and workload types."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    workload_types = list(comparison_results.keys())
    strategies = list(comparison_results[workload_types[0]].keys())
    
    x = np.arange(len(workload_types))
    width = 0.8 / len(strategies)
    
    for i, strategy in enumerate(strategies):
        costs = []
        cost_stds = []
        
        for workload_type in workload_types:
            results = comparison_results[workload_type][strategy]
            cost = results["summary"]["avg_cost"]
            cost_std = results["summary"]["std_cost"]
            costs.append(cost)
            cost_stds.append(cost_std)
        
        ax.bar(x + i * width, costs, width, label=strategy, alpha=0.8, yerr=cost_stds, capsize=5)
    
    ax.set_xlabel("Workload Type")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Cost Comparison Across Strategies and Workload Types")
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(workload_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cost_comparison.png"), dpi=150)
    plt.close()

def _plot_sla_comparison(comparison_results: Dict[str, Any], save_dir: str):
    """Plot SLA violation rate comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    workload_types = list(comparison_results.keys())
    strategies = list(comparison_results[workload_types[0]].keys())
    
    x = np.arange(len(workload_types))
    width = 0.8 / len(strategies)
    
    for i, strategy in enumerate(strategies):
        violation_rates = []
        rate_stds = []
        
        for workload_type in workload_types:
            results = comparison_results[workload_type][strategy]
            rate = results["summary"]["sla_violation_rate"]
            rate_std = results["summary"]["std_violations"] / 300  # Convert to rate
            violation_rates.append(rate)
            rate_stds.append(rate_std)
        
        ax.bar(x + i * width, violation_rates, width, label=strategy, alpha=0.8, yerr=rate_stds, capsize=5)
    
    ax.set_xlabel("Workload Type")
    ax.set_ylabel("SLA Violation Rate")
    ax.set_title("SLA Violation Rate Comparison")
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(workload_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sla_comparison.png"), dpi=150)
    plt.close()

def _plot_service_usage(comparison_results: Dict[str, Any], save_dir: str):
    """Plot service usage patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    workload_types = list(comparison_results.keys())
    strategies = list(comparison_results[workload_types[0]].keys())
    
    for i, workload_type in enumerate(workload_types):
        ax = axes[i]
        
        for strategy in strategies:
            results = comparison_results[workload_type][strategy]
            service_usage = results["service_usage"]
            
            services = list(service_usage.keys())
            usage_values = [np.mean(service_usage[service]) for service in services]
            
            ax.bar(services, usage_values, alpha=0.7, label=strategy)
        
        ax.set_title(f"Service Usage - {workload_type.title()}")
        ax.set_ylabel("Average Instances")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "service_usage.png"), dpi=150)
    plt.close()

def _plot_performance_tradeoffs(comparison_results: Dict[str, Any], save_dir: str):
    """Plot performance trade-offs (cost vs SLA violations)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_results)))
    
    for i, (workload_type, workload_results) in enumerate(comparison_results.items()):
        for strategy_name, results in workload_results.items():
            cost = results["summary"]["avg_cost"]
            sla_rate = results["summary"]["sla_violation_rate"]
            
            ax.scatter(cost, sla_rate, 
                      label=f"{strategy_name} ({workload_type})",
                      color=colors[i], s=100, alpha=0.7)
    
    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("SLA Violation Rate")
    ax.set_title("Performance Trade-offs: Cost vs SLA Violations")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_tradeoffs.png"), dpi=150)
    plt.close()

# Test the comparison utilities
if __name__ == "__main__":
    # Test with a simple comparison
    print("Testing comparison utilities...")
    
    # Create test strategies
    strategies = {
        "cost_optimized": create_agent("cost_optimized"),
        "reliability_optimized": create_agent("reliability_optimized"),
        "hybrid": create_agent("hybrid")
    }
    
    # Run comparison
    comparison_results = compare_strategies(
        strategies=strategies,
        workload_types=["diurnal", "steady"],
        n_episodes=3
    )
    
    # Generate report
    report = generate_comparison_report(comparison_results)
    print(report)
    
    # Generate plots
    plot_comparison_results(comparison_results)
    
    print("Comparison utilities test completed!")
