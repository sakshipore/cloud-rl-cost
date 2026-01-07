# visualization/academic_plots.py
"""
Academic-Quality Visualization Module

This module provides publication-ready visualizations for:
- Cost comparison across approaches
- SLA analysis (violation rates, latency, availability)
- Adaptive service selection patterns
- Reward analysis
- Performance trade-offs (Pareto frontiers)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os


def plot_cost_comparison(results: Dict[str, Dict[str, Any]], 
                        save_path: Optional[str] = None,
                        title: str = "Cost Comparison Across Approaches") -> None:
    """
    Plot bar chart comparing total costs across all approaches.
    
    Args:
        results: Comparison results dictionary
        save_path: Path to save plot
        title: Plot title
    """
    approaches = []
    costs = []
    std_costs = []
    
    for approach_name, metrics in results.items():
        # Skip individual traditional approaches if we have aggregated "traditional"
        if approach_name.startswith("traditional_") and "traditional" in results:
            continue
            
        approaches.append(approach_name.replace("_", " ").title())
        costs.append(metrics.get("total_cost", 0))
        std_costs.append(metrics.get("std_cost", 0))
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(approaches))
    bars = ax.bar(x_pos, costs, yerr=std_costs, capsize=5, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Color bars by approach type
    colors = []
    for approach in approaches:
        if "Traditional" in approach:
            colors.append('#1f77b4')  # Blue
        elif "Vpq" in approach or "VpQ" in approach:
            colors.append('#ff7f0e')  # Orange
        elif "Dqn" in approach or "DQN" in approach:
            colors.append('#2ca02c')  # Green
        else:
            colors.append('#d62728')  # Red
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Approach", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Cost ($)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Traditional Heuristic'),
        Patch(facecolor='#ff7f0e', label='VpQ-inspired RL'),
        Patch(facecolor='#2ca02c', label='DQN-based Adaptive')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sla_comparison(results: Dict[str, Dict[str, Any]],
                        save_path: Optional[str] = None,
                        title: str = "SLA Compliance Comparison") -> None:
    """
    Plot comparison of SLA metrics (violation rate, availability, latency).
    
    Args:
        results: Comparison results dictionary
        save_path: Path to save plot
        title: Plot title
    """
    approaches = []
    sla_rates = []
    availabilities = []
    latencies = []
    
    for approach_name, metrics in results.items():
        # Skip individual traditional approaches if we have aggregated "traditional"
        if approach_name.startswith("traditional_") and "traditional" in results:
            continue
            
        approaches.append(approach_name.replace("_", " ").title())
        sla_rates.append(metrics.get("sla_violation_rate", 0) * 100)  # Convert to percentage
        availabilities.append(metrics.get("overall_availability", 0) * 100)
        latencies.append(metrics.get("avg_latency", 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    x_pos = np.arange(len(approaches))
    width = 0.6
    
    # SLA Violation Rate
    axes[0].bar(x_pos, sla_rates, width, alpha=0.8, color='#d62728', 
                edgecolor='black', linewidth=1)
    axes[0].set_ylabel("SLA Violation Rate (%)", fontsize=11, fontweight='bold')
    axes[0].set_title("SLA Violation Rate", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sla_rates):
        axes[0].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Availability
    axes[1].bar(x_pos, availabilities, width, alpha=0.8, color='#2ca02c',
                edgecolor='black', linewidth=1)
    axes[1].set_ylabel("Overall Availability (%)", fontsize=11, fontweight='bold')
    axes[1].set_title("Service Availability", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(availabilities):
        axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Average Latency
    axes[2].bar(x_pos, latencies, width, alpha=0.8, color='#ff7f0e',
                edgecolor='black', linewidth=1)
    axes[2].set_ylabel("Average Latency (ms)", fontsize=11, fontweight='bold')
    axes[2].set_title("Average Latency", fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(latencies):
        axes[2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_performance_tradeoffs(results: Dict[str, Dict[str, Any]],
                               save_path: Optional[str] = None,
                               title: str = "Cost vs SLA Violation Trade-off") -> None:
    """
    Plot scatter plot showing cost vs SLA violation trade-off (Pareto frontier).
    
    Args:
        results: Comparison results dictionary
        save_path: Path to save plot
        title: Plot title
    """
    costs = []
    sla_rates = []
    labels = []
    colors = []
    
    for approach_name, metrics in results.items():
        # Skip individual traditional approaches if we have aggregated "traditional"
        if approach_name.startswith("traditional_") and "traditional" in results:
            continue
            
        costs.append(metrics.get("total_cost", 0))
        sla_rates.append(metrics.get("sla_violation_rate", 0) * 100)
        labels.append(approach_name.replace("_", " ").title())
        
        # Color by approach type
        if "traditional" in approach_name.lower():
            colors.append('#1f77b4')
        elif "vpq" in approach_name.lower():
            colors.append('#ff7f0e')
        elif "dqn" in approach_name.lower():
            colors.append('#2ca02c')
        else:
            colors.append('#d62728')
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    for i, (cost, sla, label, color) in enumerate(zip(costs, sla_rates, labels, colors)):
        ax.scatter(cost, sla, s=200, alpha=0.7, color=color, edgecolors='black', 
                  linewidth=1.5, label=label if i < 3 else None)
        ax.annotate(label, (cost, sla), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Total Cost ($)", fontsize=12, fontweight='bold')
    ax.set_ylabel("SLA Violation Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax.axvline(x=np.mean(costs), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(sla_rates), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_service_selection_frequency(history: Dict[str, Any],
                                     save_path: Optional[str] = None,
                                     title: str = "Service Selection Frequency Over Time") -> None:
    """
    Plot service selection frequency over time.
    
    Args:
        history: Environment history dictionary
        save_path: Path to save plot
        title: Plot title
    """
    if "service_usage" not in history:
        raise ValueError("History must contain 'service_usage' key")
    
    service_usage = history["service_usage"]
    time_steps = np.arange(1, len(list(service_usage.values())[0]) + 1)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (service_name, usage) in enumerate(service_usage.items()):
        ax.plot(time_steps, usage, label=service_name.replace("_", " ").title(),
               linewidth=2, alpha=0.8, color=colors[i % len(colors)])
    
    ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Instances", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cost_over_time(histories: Dict[str, Dict[str, Any]],
                       save_path: Optional[str] = None,
                       title: str = "Cost Over Time Comparison") -> None:
    """
    Plot cumulative cost over time for different approaches.
    
    Args:
        histories: Dictionary mapping approach names to history dictionaries
        save_path: Path to save plot
        title: Plot title
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'traditional': '#1f77b4', 'vpq_inspired': '#ff7f0e', 'dqn': '#2ca02c'}
    
    for approach_name, history in histories.items():
        if "total_cost" in history:
            costs = history["total_cost"]
            cumulative_costs = np.cumsum(costs)
            time_steps = np.arange(1, len(cumulative_costs) + 1)
            
            color = colors.get(approach_name.split('_')[0], '#d62728')
            ax.plot(time_steps, cumulative_costs, label=approach_name.replace("_", " ").title(),
                   linewidth=2, alpha=0.8, color=color)
    
    ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Cost ($)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comprehensive_comparison(results: Dict[str, Dict[str, Any]],
                                 output_dir: str = "outputs",
                                 prefix: str = "comparison") -> Dict[str, str]:
    """
    Generate all comprehensive comparison plots.
    
    Args:
        results: Comparison results dictionary
        output_dir: Output directory
        prefix: Filename prefix
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Cost comparison
    plot_paths['cost_comparison'] = os.path.join(output_dir, f"{prefix}_cost.png")
    plot_cost_comparison(results, save_path=plot_paths['cost_comparison'])
    
    # SLA comparison
    plot_paths['sla_comparison'] = os.path.join(output_dir, f"{prefix}_sla.png")
    plot_sla_comparison(results, save_path=plot_paths['sla_comparison'])
    
    # Performance trade-offs
    plot_paths['tradeoffs'] = os.path.join(output_dir, f"{prefix}_tradeoffs.png")
    plot_performance_tradeoffs(results, save_path=plot_paths['tradeoffs'])
    
    return plot_paths


# Test the visualization functions
if __name__ == "__main__":
    # Create sample results for testing
    sample_results = {
        "traditional_cost_optimized": {
            "total_cost": 1200.0,
            "std_cost": 50.0,
            "sla_violation_rate": 0.085,
            "overall_availability": 0.95,
            "avg_latency": 220.0
        },
        "vpq_inspired": {
            "total_cost": 950.0,
            "std_cost": 40.0,
            "sla_violation_rate": 0.12,
            "overall_availability": 0.92,
            "avg_latency": 250.0
        },
        "dqn": {
            "total_cost": 850.0,
            "std_cost": 30.0,
            "sla_violation_rate": 0.021,
            "overall_availability": 0.99,
            "avg_latency": 195.0
        }
    }
    
    # Generate plots
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating academic-quality plots...")
    plot_paths = plot_comprehensive_comparison(sample_results, output_dir)
    
    print("Generated plots:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

