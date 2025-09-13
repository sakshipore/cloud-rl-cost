# rl/evaluate.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from stable_baselines3 import DQN
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent, RuleBasedAgent

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for comparing RL and rule-based strategies.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics to track
        self.metrics = [
            "total_cost",
            "sla_violations", 
            "sla_violation_rate",
            "total_interruptions",
            "avg_latency",
            "max_latency",
            "resource_efficiency",
            "cost_per_request",
            "service_utilization"
        ]
    
    def evaluate_strategy(self, strategy, env: EnhancedCloudCostGym, 
                         n_episodes: int = 10, strategy_name: str = "Unknown") -> Dict[str, Any]:
        """
        Evaluate a strategy (RL model or rule-based agent) on the environment.
        
        Args:
            strategy: Strategy to evaluate (RL model or rule-based agent)
            env: Environment to evaluate on
            n_episodes: Number of episodes to run
            strategy_name: Name of the strategy for logging
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {strategy_name}...")
        
        results = {
            "strategy_name": strategy_name,
            "episodes": [],
            "metrics": {metric: [] for metric in self.metrics},
            "service_usage": {name: [] for name in env.services.keys()},
            "service_costs": {name: [] for name in env.services.keys()},
            "action_distribution": {"service_selection": [], "scaling_actions": []}
        }
        
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            episode_metrics = {metric: 0.0 for metric in self.metrics}
            episode_service_usage = {name: [] for name in env.services.keys()}
            episode_service_costs = {name: [] for name in env.services.keys()}
            episode_actions = {"service_selection": [], "scaling_actions": []}
            
            while not done:
                # Get action from strategy
                if isinstance(strategy, RuleBasedAgent):
                    action = strategy.predict(obs)
                else:  # RL model
                    action, _ = strategy.predict(obs, deterministic=True)
                
                # Convert combined action to service and scale components
                service_type = action // 3
                scale_action = action % 3
                
                # Store action distribution
                episode_actions["service_selection"].append(service_type)
                episode_actions["scaling_actions"].append(scale_action)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update metrics
                episode_metrics["total_cost"] += -reward  # Reward is negative cost
                episode_metrics["sla_violations"] += info.get("sla_violation", 0)
                episode_metrics["total_interruptions"] += info.get("interrupted", 0)
                
                # Update service usage and costs
                for service_name in env.services.keys():
                    episode_service_usage[service_name].append(
                        env.service_instances[service_name]
                    )
                    episode_service_costs[service_name].append(
                        info.get("service_costs", {}).get(service_name, 0.0)
                    )
            
            # Calculate final episode metrics
            total_requests = sum(env.history["demand"])
            episode_metrics["sla_violation_rate"] = (
                episode_metrics["sla_violations"] / len(env.history["demand"])
            )
            episode_metrics["avg_latency"] = np.mean(env.history["latency"])
            episode_metrics["max_latency"] = np.max(env.history["latency"])
            episode_metrics["cost_per_request"] = (
                episode_metrics["total_cost"] / total_requests if total_requests > 0 else 0
            )
            
            # Calculate resource efficiency
            total_capacity = sum([
                sum(episode_service_usage[service_name]) * env.services[service_name].capacity
                for service_name in env.services.keys()
            ])
            episode_metrics["resource_efficiency"] = (
                total_requests / total_capacity if total_capacity > 0 else 0
            )
            
            # Calculate service utilization
            service_utilization = {}
            for service_name in env.services.keys():
                avg_instances = np.mean(episode_service_usage[service_name])
                service_utilization[service_name] = avg_instances
            episode_metrics["service_utilization"] = service_utilization
            
            # Store episode results
            results["episodes"].append(episode)
            for metric in self.metrics:
                if metric == "service_utilization":
                    results["metrics"][metric].append(episode_metrics[metric])
                else:
                    results["metrics"][metric].append(episode_metrics[metric])
            
            # Store service usage and costs
            for service_name in env.services.keys():
                results["service_usage"][service_name].append(
                    np.mean(episode_service_usage[service_name])
                )
                results["service_costs"][service_name].append(
                    sum(episode_service_costs[service_name])
                )
            
            # Store action distribution
            results["action_distribution"]["service_selection"].extend(
                episode_actions["service_selection"]
            )
            results["action_distribution"]["scaling_actions"].extend(
                episode_actions["scaling_actions"]
            )
        
        # Calculate summary statistics
        results["summary"] = self._calculate_summary_stats(results)
        
        return results
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the results."""
        summary = {}
        
        for metric in self.metrics:
            if metric == "service_utilization":
                # Handle service utilization separately
                summary[metric] = {}
                for service_name in results["metrics"][metric][0].keys():
                    values = [ep[service_name] for ep in results["metrics"][metric]]
                    summary[metric][service_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
            else:
                values = results["metrics"][metric]
                summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return summary
    
    def compare_strategies(self, strategies: Dict[str, Any], 
                          workload_types: List[str] = None,
                          n_episodes: int = 10) -> Dict[str, Any]:
        """
        Compare multiple strategies across different workload types.
        
        Args:
            strategies: Dictionary mapping strategy names to strategy objects
            workload_types: List of workload types to test
            n_episodes: Number of episodes per strategy per workload
            
        Returns:
            Dictionary with comparison results
        """
        if workload_types is None:
            workload_types = ["diurnal", "steady", "batch", "bursty"]
        
        comparison_results = {}
        
        for workload_type in workload_types:
            print(f"\nEvaluating strategies on {workload_type} workload...")
            
            # Create environment for this workload type
            env = EnhancedCloudCostGym(n_steps=300, seed=42, workload_type=workload_type)
            
            workload_results = {}
            
            for strategy_name, strategy in strategies.items():
                results = self.evaluate_strategy(strategy, env, n_episodes, strategy_name)
                workload_results[strategy_name] = results
            
            comparison_results[workload_type] = workload_results
        
        return comparison_results
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any], 
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
            metrics = ["total_cost", "sla_violation_rate", "avg_latency", "resource_efficiency"]
            
            # Header
            header = f"{'Strategy':<20}"
            for metric in metrics:
                header += f"{metric.replace('_', ' ').title():<20}"
            report.append(header)
            report.append("-" * len(header))
            
            # Data rows
            for strategy_name in strategies:
                results = workload_results[strategy_name]
                row = f"{strategy_name:<20}"
                
                for metric in metrics:
                    if metric in results["summary"]:
                        value = results["summary"][metric]["mean"]
                        if metric == "total_cost":
                            row += f"${value:.2f}".ljust(20)
                        elif metric == "sla_violation_rate":
                            row += f"{value:.1%}".ljust(20)
                        elif metric == "avg_latency":
                            row += f"{value:.1f}ms".ljust(20)
                        else:
                            row += f"{value:.3f}".ljust(20)
                    else:
                        row += "N/A".ljust(20)
                
                report.append(row)
            
            report.append("")
        
        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        
        # Calculate overall best performers
        overall_metrics = {metric: {} for metric in metrics}
        
        for workload_type, workload_results in comparison_results.items():
            for strategy_name, results in workload_results.items():
                for metric in metrics:
                    if metric in results["summary"]:
                        value = results["summary"][metric]["mean"]
                        if strategy_name not in overall_metrics[metric]:
                            overall_metrics[metric][strategy_name] = []
                        overall_metrics[metric][strategy_name].append(value)
        
        for metric in metrics:
            if overall_metrics[metric]:
                # Find best strategy (lowest for cost, highest for efficiency)
                if metric in ["total_cost", "sla_violation_rate", "avg_latency"]:
                    best_strategy = min(overall_metrics[metric].keys(), 
                                      key=lambda x: np.mean(overall_metrics[metric][x]))
                else:
                    best_strategy = max(overall_metrics[metric].keys(), 
                                      key=lambda x: np.mean(overall_metrics[metric][x]))
                
                report.append(f"Best {metric.replace('_', ' ').title()}: {best_strategy}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text
    
    def plot_comparison_results(self, comparison_results: Dict[str, Any], 
                               save_dir: str = None):
        """
        Generate comparison plots.
        
        Args:
            comparison_results: Results from compare_strategies
            save_dir: Directory to save plots
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot 1: Cost comparison across workload types
        self._plot_cost_comparison(comparison_results, save_dir)
        
        # Plot 2: SLA violation rate comparison
        self._plot_sla_comparison(comparison_results, save_dir)
        
        # Plot 3: Service usage patterns
        self._plot_service_usage(comparison_results, save_dir)
        
        # Plot 4: Performance trade-offs
        self._plot_performance_tradeoffs(comparison_results, save_dir)
    
    def _plot_cost_comparison(self, comparison_results: Dict[str, Any], save_dir: str):
        """Plot cost comparison across strategies and workload types."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        workload_types = list(comparison_results.keys())
        strategies = list(comparison_results[workload_types[0]].keys())
        
        x = np.arange(len(workload_types))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            costs = []
            for workload_type in workload_types:
                if strategy in comparison_results[workload_type]:
                    results = comparison_results[workload_type][strategy]
                    cost = results["summary"]["total_cost"]["mean"]
                    costs.append(cost)
                else:
                    costs.append(0)
            
            ax.bar(x + i * width, costs, width, label=strategy, alpha=0.8)
        
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
    
    def _plot_sla_comparison(self, comparison_results: Dict[str, Any], save_dir: str):
        """Plot SLA violation rate comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        workload_types = list(comparison_results.keys())
        strategies = list(comparison_results[workload_types[0]].keys())
        
        x = np.arange(len(workload_types))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            violation_rates = []
            for workload_type in workload_types:
                results = comparison_results[workload_type][strategy]
                rate = results["summary"]["sla_violation_rate"]["mean"]
                violation_rates.append(rate)
            
            ax.bar(x + i * width, violation_rates, width, label=strategy, alpha=0.8)
        
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
    
    def _plot_service_usage(self, comparison_results: Dict[str, Any], save_dir: str):
        """Plot service usage patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        workload_types = list(comparison_results.keys())
        strategies = list(comparison_results[workload_types[0]].keys())
        
        for i, workload_type in enumerate(workload_types):
            ax = axes[i]
            
            for strategy in strategies:
                results = comparison_results[workload_type][strategy]
                service_usage = results["summary"]["service_utilization"]
                
                services = list(service_usage.keys())
                usage_values = [service_usage[service]["mean"] for service in services]
                
                ax.bar(services, usage_values, alpha=0.7, label=strategy)
            
            ax.set_title(f"Service Usage - {workload_type.title()}")
            ax.set_ylabel("Average Instances")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "service_usage.png"), dpi=150)
        plt.close()
    
    def _plot_performance_tradeoffs(self, comparison_results: Dict[str, Any], save_dir: str):
        """Plot performance trade-offs (cost vs SLA violations)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_results)))
        
        for i, (workload_type, workload_results) in enumerate(comparison_results.items()):
            for strategy_name, results in workload_results.items():
                cost = results["summary"]["total_cost"]["mean"]
                sla_rate = results["summary"]["sla_violation_rate"]["mean"]
                
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


# Test the evaluation framework
if __name__ == "__main__":
    # Create evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Test with a simple comparison
    print("Testing evaluation framework...")
    
    # Create test strategies
    strategies = {
        "cost_optimized": create_agent("cost_optimized"),
        "reliability_optimized": create_agent("reliability_optimized"),
        "hybrid": create_agent("hybrid")
    }
    
    # Run comparison
    comparison_results = evaluator.compare_strategies(
        strategies=strategies,
        workload_types=["diurnal", "steady"],
        n_episodes=3
    )
    
    # Generate report
    report = evaluator.generate_comparison_report(comparison_results)
    print(report)
    
    # Generate plots
    evaluator.plot_comparison_results(comparison_results)
    
    print("Evaluation framework test completed!")
