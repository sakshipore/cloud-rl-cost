# run_experiments.py
"""
Comprehensive experiment runner for cloud RL cost optimization.
This script runs the complete pipeline: training, evaluation, and comparison.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from stable_baselines3 import DQN

# Import our modules
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent
from baselines.compare import compare_strategies, generate_comparison_report, plot_comparison_results
from rl.evaluate import ComprehensiveEvaluator

def run_complete_experiment(workload_types: List[str] = None,
                           n_episodes: int = 10,
                           n_steps: int = 300,
                           total_timesteps: int = 20000,
                           output_dir: str = "outputs",
                           seed: int = 42) -> Dict[str, Any]:
    """
    Run complete experiment pipeline.
    
    Args:
        workload_types: List of workload types to test
        n_episodes: Number of episodes per strategy per workload
        n_steps: Number of steps per episode
        total_timesteps: Total training timesteps for RL
        output_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary with all experiment results
    """
    if workload_types is None:
        workload_types = ["diurnal", "steady", "batch", "bursty"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CLOUD RL COST OPTIMIZATION - COMPLETE EXPERIMENT")
    print("=" * 80)
    print(f"Workload types: {workload_types}")
    print(f"Episodes per strategy: {n_episodes}")
    print(f"Steps per episode: {n_steps}")
    print(f"RL training timesteps: {total_timesteps}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Step 1: Train RL models for each workload type
    print("\nSTEP 1: Training RL Models")
    print("-" * 40)
    
    rl_models = {}
    
    for workload_type in workload_types:
        print(f"\nTraining DQN on {workload_type} workload...")
        
        # Create training environment
        train_env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed, workload_type=workload_type)
        
        # Create DQN model
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=0,  # Reduce output for cleaner logs
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05
        )
        
        # Train the model
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Save the model
        model_path = os.path.join(output_dir, f"dqn_model_{workload_type}")
        model.save(model_path)
        
        rl_models[workload_type] = model
        print(f"  Model trained and saved to {model_path}")
    
    # Step 2: Create rule-based baselines
    print("\nSTEP 2: Setting up Rule-based Baselines")
    print("-" * 40)
    
    baseline_strategies = {
        "cost_optimized": create_agent("cost_optimized"),
        "reliability_optimized": create_agent("reliability_optimized"),
        "hybrid": create_agent("hybrid"),
        "workload_aware": create_agent("workload_aware"),
        "threshold": create_agent("threshold")
    }
    
    print(f"Created {len(baseline_strategies)} rule-based strategies")
    
    # Step 3: Run comprehensive evaluation
    print("\nSTEP 3: Running Comprehensive Evaluation")
    print("-" * 40)
    
    evaluator = ComprehensiveEvaluator(output_dir)
    
    all_results = {}
    
    for workload_type in workload_types:
        print(f"\nEvaluating strategies on {workload_type} workload...")
        
        # Create environment for this workload type
        env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed, workload_type=workload_type)
        
        # Combine RL and baseline strategies
        strategies = baseline_strategies.copy()
        strategies[f"rl_{workload_type}"] = rl_models[workload_type]
        
        # Run evaluation
        workload_results = evaluator.compare_strategies(
            strategies=strategies,
            workload_types=[workload_type],
            n_episodes=n_episodes
        )
        
        all_results[workload_type] = workload_results[workload_type]
    
    # Step 4: Generate comprehensive comparison
    print("\nSTEP 4: Generating Comprehensive Comparison")
    print("-" * 40)
    
    # Create cross-workload comparison
    cross_workload_results = {}
    
    for workload_type in workload_types:
        cross_workload_results[workload_type] = all_results[workload_type]
    
    # Generate comparison report
    report = evaluator.generate_comparison_report(
        cross_workload_results,
        save_path=os.path.join(output_dir, "comprehensive_comparison_report.txt")
    )
    
    print("Comparison report generated")
    
    # Generate plots
    evaluator.plot_comparison_results(cross_workload_results, output_dir)
    print("Comparison plots generated")
    
    # Step 5: Save detailed results
    print("\nSTEP 5: Saving Detailed Results")
    print("-" * 40)
    
    # Save all results to JSON
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save summary statistics
    summary_stats = _generate_summary_statistics(all_results)
    with open(os.path.join(output_dir, "summary_statistics.json"), "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print("Detailed results saved")
    
    # Step 6: Print final summary
    print("\nSTEP 6: Final Summary")
    print("-" * 40)
    
    _print_final_summary(summary_stats)
    
    return all_results

def _generate_summary_statistics(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from all results."""
    summary = {
        "workload_types": list(all_results.keys()),
        "strategies": {},
        "best_performers": {},
        "overall_metrics": {}
    }
    
    # Get all unique strategies
    all_strategies = set()
    for workload_results in all_results.values():
        all_strategies.update(workload_results.keys())
    
    # Calculate overall metrics for each strategy
    for strategy in all_strategies:
        strategy_metrics = {
            "total_cost": [],
            "sla_violation_rate": [],
            "avg_latency": [],
            "resource_efficiency": []
        }
        
        for workload_type, workload_results in all_results.items():
            if strategy in workload_results:
                results = workload_results[strategy]
                summary_metrics = results["summary"]
                
                strategy_metrics["total_cost"].append(summary_metrics["total_cost"]["mean"])
                strategy_metrics["sla_violation_rate"].append(summary_metrics["sla_violation_rate"]["mean"])
                strategy_metrics["avg_latency"].append(summary_metrics["avg_latency"]["mean"])
                strategy_metrics["resource_efficiency"].append(summary_metrics["resource_efficiency"]["mean"])
        
        # Calculate averages across workload types
        summary["strategies"][strategy] = {
            "avg_cost": np.mean(strategy_metrics["total_cost"]),
            "avg_sla_rate": np.mean(strategy_metrics["sla_violation_rate"]),
            "avg_latency": np.mean(strategy_metrics["avg_latency"]),
            "avg_efficiency": np.mean(strategy_metrics["resource_efficiency"]),
            "cost_std": np.std(strategy_metrics["total_cost"]),
            "sla_rate_std": np.std(strategy_metrics["sla_violation_rate"])
        }
    
    # Find best performers
    summary["best_performers"] = {
        "lowest_cost": min(summary["strategies"].keys(), 
                          key=lambda x: summary["strategies"][x]["avg_cost"]),
        "lowest_sla_rate": min(summary["strategies"].keys(), 
                              key=lambda x: summary["strategies"][x]["avg_sla_rate"]),
        "highest_efficiency": max(summary["strategies"].keys(), 
                                 key=lambda x: summary["strategies"][x]["avg_efficiency"])
    }
    
    return summary

def _print_final_summary(summary_stats: Dict[str, Any]):
    """Print final summary of results."""
    print("\nFINAL RESULTS SUMMARY")
    print("=" * 50)
    
    print("\nBest Performers:")
    best = summary_stats["best_performers"]
    print(f"  Lowest Cost: {best['lowest_cost']}")
    print(f"  Lowest SLA Rate: {best['lowest_sla_rate']}")
    print(f"  Highest Efficiency: {best['highest_efficiency']}")
    
    print("\nStrategy Performance (Average across all workloads):")
    print(f"{'Strategy':<25} {'Avg Cost':<12} {'SLA Rate':<12} {'Efficiency':<12}")
    print("-" * 65)
    
    for strategy, metrics in summary_stats["strategies"].items():
        print(f"{strategy:<25} "
              f"${metrics['avg_cost']:.2f}".ljust(12) +
              f"{metrics['avg_sla_rate']:.1%}".ljust(12) +
              f"{metrics['avg_efficiency']:.3f}".ljust(12))
    
    print("\n" + "=" * 50)

def main():
    """Main function to run experiments from command line."""
    parser = argparse.ArgumentParser(description="Run cloud RL cost optimization experiments")
    
    parser.add_argument("--workload-types", nargs="+", 
                       default=["diurnal", "steady", "batch", "bursty"],
                       help="Workload types to test")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per strategy per workload")
    parser.add_argument("--steps", type=int, default=300,
                       help="Number of steps per episode")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Total training timesteps for RL")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Run the complete experiment
    results = run_complete_experiment(
        workload_types=args.workload_types,
        n_episodes=args.episodes,
        n_steps=args.steps,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print(f"\nExperiment completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
