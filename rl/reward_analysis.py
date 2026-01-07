# rl/reward_analysis.py
"""
Reward analysis and visualization module for academic-quality plots.

This module provides functions to analyze and visualize reward function components,
including reward vs time, reward vs workload intensity, and reward component breakdown.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os


def plot_reward_vs_time(history: Dict[str, Any], save_path: Optional[str] = None,
                        title: str = "Reward vs Time") -> None:
    """
    Plot reward over time with academic-quality formatting.
    
    Args:
        history: Environment history dictionary containing reward_components
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if "reward_components" not in history:
        raise ValueError("History must contain 'reward_components' key")
    
    reward_components = history["reward_components"]
    total_rewards = reward_components.get("total_reward", [])
    
    if not total_rewards:
        raise ValueError("No reward data found in history")
    
    time_steps = np.arange(1, len(total_rewards) + 1)
    
    # Apply moving average for smoother visualization
    window_size = max(5, min(25, len(total_rewards) // 20))
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        smoothed_rewards = np.convolve(total_rewards, kernel, mode='same')
    else:
        smoothed_rewards = total_rewards
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw and smoothed rewards
    ax.plot(time_steps, total_rewards, alpha=0.3, color='#1f77b4', linewidth=0.5, label='Raw Reward')
    ax.plot(time_steps, smoothed_rewards, color='#1f77b4', linewidth=2, label='Smoothed Reward (Moving Average)')
    
    # Add horizontal line at zero for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    textstr = f'Mean: {mean_reward:.2f}\nStd: {std_reward:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reward_vs_workload_intensity(history: Dict[str, Any], 
                                      save_path: Optional[str] = None,
                                      title: str = "Reward vs Workload Intensity") -> None:
    """
    Plot reward as a function of workload intensity (demand).
    
    Args:
        history: Environment history dictionary
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if "demand" not in history or "reward_components" not in history:
        raise ValueError("History must contain 'demand' and 'reward_components' keys")
    
    demands = np.array(history["demand"])
    total_rewards = np.array(history["reward_components"].get("total_reward", []))
    
    if len(demands) != len(total_rewards):
        raise ValueError("Demand and reward arrays must have the same length")
    
    # Bin demands for better visualization
    demand_bins = np.linspace(demands.min(), demands.max(), 20)
    bin_indices = np.digitize(demands, demand_bins)
    
    # Calculate mean reward for each demand bin
    bin_rewards = []
    bin_demands = []
    for i in range(1, len(demand_bins)):
        mask = bin_indices == i
        if np.any(mask):
            bin_rewards.append(np.mean(total_rewards[mask]))
            bin_demands.append(np.mean(demands[mask]))
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot with binned averages
    ax.scatter(demands, total_rewards, alpha=0.2, s=10, color='#1f77b4', label='Individual Steps')
    ax.plot(bin_demands, bin_rewards, color='#d62728', linewidth=2.5, marker='o', 
            markersize=6, label='Binned Average')
    
    ax.set_xlabel("Workload Intensity (Requests/sec)", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(demands, total_rewards)[0, 1]
    textstr = f'Correlation: {correlation:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reward_component_breakdown(history: Dict[str, Any],
                                   save_path: Optional[str] = None,
                                   title: str = "Reward Component Breakdown") -> None:
    """
    Plot stacked area chart showing reward component breakdown over time.
    
    Args:
        history: Environment history dictionary
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if "reward_components" not in history:
        raise ValueError("History must contain 'reward_components' key")
    
    reward_components = history["reward_components"]
    
    cost_term = np.array(reward_components.get("cost_term", []))
    latency_penalty = np.array(reward_components.get("latency_penalty", []))
    availability_penalty = np.array(reward_components.get("availability_penalty", []))
    high_load_bonus = np.array(reward_components.get("high_load_bonus", []))
    
    if len(cost_term) == 0:
        raise ValueError("No reward component data found")
    
    time_steps = np.arange(1, len(cost_term) + 1)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Stack components (note: penalties are negative, so we stack from bottom)
    # Start with cost term (negative)
    bottom = cost_term
    
    # Add latency penalty (negative, so it goes further down)
    ax.fill_between(time_steps, bottom, bottom + latency_penalty, 
                     alpha=0.7, color='#ff7f0e', label='Latency Penalty')
    bottom = bottom + latency_penalty
    
    # Add availability penalty (negative, so it goes further down)
    ax.fill_between(time_steps, bottom, bottom + availability_penalty,
                     alpha=0.7, color='#d62728', label='Availability Penalty')
    bottom = bottom + availability_penalty
    
    # Add high load bonus (positive, so it goes up)
    ax.fill_between(time_steps, bottom, bottom + high_load_bonus,
                     alpha=0.7, color='#2ca02c', label='High Load Bonus')
    
    # Plot cost term line
    ax.plot(time_steps, cost_term, color='#1f77b4', linewidth=2, 
            label='Cost Term', alpha=0.8)
    
    # Add zero reference line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Reward Component Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(frameon=False, fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    total_cost = np.sum(cost_term)
    total_latency_penalty = np.sum(latency_penalty)
    total_availability_penalty = np.sum(availability_penalty)
    total_bonus = np.sum(high_load_bonus)
    
    textstr = (f'Total Cost: {total_cost:.2f}\n'
               f'Total Latency Penalty: {total_latency_penalty:.2f}\n'
               f'Total Availability Penalty: {total_availability_penalty:.2f}\n'
               f'Total Bonus: {total_bonus:.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_reward_component_comparison(history: Dict[str, Any],
                                    save_path: Optional[str] = None,
                                    title: str = "Reward Component Comparison") -> None:
    """
    Plot bar chart comparing total values of each reward component.
    
    Args:
        history: Environment history dictionary
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if "reward_components" not in history:
        raise ValueError("History must contain 'reward_components' key")
    
    reward_components = history["reward_components"]
    
    components = {
        'Cost Term': np.sum(reward_components.get("cost_term", [])),
        'Latency Penalty': np.sum(reward_components.get("latency_penalty", [])),
        'Availability Penalty': np.sum(reward_components.get("availability_penalty", [])),
        'High Load Bonus': np.sum(reward_components.get("high_load_bonus", []))
    }
    
    component_names = list(components.keys())
    component_values = list(components.values())
    
    # Color scheme: cost (blue), penalties (red/orange), bonus (green)
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(component_names, component_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, component_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Total Component Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_all_reward_plots(history: Dict[str, Any], output_dir: str,
                             prefix: str = "reward_analysis") -> Dict[str, str]:
    """
    Generate all reward analysis plots and save them to the output directory.
    
    Args:
        history: Environment history dictionary
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Generate all plots
    plot_paths['reward_vs_time'] = os.path.join(output_dir, f"{prefix}_vs_time.png")
    plot_reward_vs_time(history, save_path=plot_paths['reward_vs_time'])
    
    plot_paths['reward_vs_workload'] = os.path.join(output_dir, f"{prefix}_vs_workload.png")
    plot_reward_vs_workload_intensity(history, save_path=plot_paths['reward_vs_workload'])
    
    plot_paths['reward_breakdown'] = os.path.join(output_dir, f"{prefix}_breakdown.png")
    plot_reward_component_breakdown(history, save_path=plot_paths['reward_breakdown'])
    
    plot_paths['reward_comparison'] = os.path.join(output_dir, f"{prefix}_comparison.png")
    plot_reward_component_comparison(history, save_path=plot_paths['reward_comparison'])
    
    return plot_paths


# Test the reward analysis functions
if __name__ == "__main__":
    # Create sample history for testing
    n_steps = 200
    sample_history = {
        "demand": np.random.randint(100, 500, n_steps).tolist(),
        "reward_components": {
            "cost_term": -np.random.uniform(0.5, 2.0, n_steps).tolist(),
            "latency_penalty": -np.random.choice([0, 2.0], n_steps, p=[0.9, 0.1]).tolist(),
            "availability_penalty": -np.random.choice([0, 1.5], n_steps, p=[0.95, 0.05]).tolist(),
            "high_load_bonus": np.random.choice([0, 0.5], n_steps, p=[0.8, 0.2]).tolist(),
            "total_reward": []
        }
    }
    
    # Calculate total rewards
    cost = np.array(sample_history["reward_components"]["cost_term"])
    latency = np.array(sample_history["reward_components"]["latency_penalty"])
    availability = np.array(sample_history["reward_components"]["availability_penalty"])
    bonus = np.array(sample_history["reward_components"]["high_load_bonus"])
    sample_history["reward_components"]["total_reward"] = (cost + latency + availability + bonus).tolist()
    
    # Generate all plots
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating reward analysis plots...")
    plot_paths = generate_all_reward_plots(sample_history, output_dir)
    
    print("Generated plots:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

