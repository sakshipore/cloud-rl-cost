# evaluation/comprehensive_eval.py
"""
Comprehensive Evaluation Framework

This module provides a complete evaluation framework that:
- Trains all three approaches (Traditional, VpQ-inspired, DQN)
- Evaluates on all workload types
- Generates comprehensive comparison reports
- Creates academic-quality visualizations
- Exports results for academic use
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from stable_baselines3 import DQN
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent
from baselines.vpq_inspired import VpQInspiredAgent
from baselines.comprehensive_comparison import ComprehensiveComparator
from visualization.academic_plots import plot_comprehensive_comparison
from rl.reward_analysis import generate_all_reward_plots
from rl.adaptive_decision import AdaptiveDecisionMaker


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator for all approaches across all workload types.
    """
    
    def __init__(self, output_dir: str = "research_outputs_final"):
        """
        Initialize the comprehensive evaluator.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.comparator = ComprehensiveComparator(output_dir)
    
    def train_all_approaches(self, workload_type: str = "steady", n_steps: int = 300,
                            seed: int = 42) -> Dict[str, Any]:
        """
        Train all three approaches for a given workload type.
        
        Args:
            workload_type: Type of workload
            n_steps: Number of steps per episode
            seed: Random seed
        
        Returns:
            Dictionary with trained models/agents
        """
        print(f"\n{'='*70}")
        print(f"TRAINING ALL APPROACHES - {workload_type.upper()} WORKLOAD")
        print(f"{'='*70}\n")
        
        # Create environment
        env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed, workload_type=workload_type)
        
        trained = {}
        
        # 1. Traditional heuristic agents (no training needed)
        print("1. Traditional Heuristic Agents (no training required)")
        traditional_agents = {
            "cost_optimized": create_agent("cost_optimized"),
            "hybrid": create_agent("hybrid"),
            "reliability_optimized": create_agent("reliability_optimized"),
            "workload_aware": create_agent("workload_aware")
        }
        trained["traditional"] = traditional_agents
        print(f"   Created {len(traditional_agents)} traditional agents\n")
        
        # 2. VpQ-inspired agent
        print("2. Training VpQ-inspired RL Baseline")
        vpq_agent = VpQInspiredAgent(
            learning_rate=0.15,  # Slightly higher learning rate for faster convergence
            epsilon=0.3,  # Start with more exploration
            epsilon_decay=0.995,
            min_epsilon=0.05
        )
        # Train for more episodes to ensure proper learning
        vpq_agent.train(env, n_episodes=200, max_steps_per_episode=n_steps)
        trained["vpq_inspired"] = vpq_agent
        print("   VpQ-inspired agent trained\n")
        
        # 3. DQN model
        print("3. Training DQN-based Adaptive Agent")
        dqn_model = DQN(
            "MlpPolicy",
            env,
            verbose=0,
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
        # Train for significantly more timesteps to ensure proper learning
        dqn_model.learn(total_timesteps=30000, progress_bar=True)
        trained["dqn"] = dqn_model
        print("   DQN model trained\n")
        
        # Save trained models
        model_dir = os.path.join(self.output_dir, "models", workload_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save VpQ agent
        vpq_path = os.path.join(model_dir, "vpq_agent.pkl")
        vpq_agent.save(vpq_path)
        
        # Save DQN model
        dqn_path = os.path.join(model_dir, "dqn_model")
        dqn_model.save(dqn_path)
        
        print(f"Models saved to {model_dir}\n")
        
        return trained
    
    def evaluate_all_approaches(self, trained: Dict[str, Any], workload_type: str = "steady",
                               n_steps: int = 300, n_episodes: int = 5,
                               seed: int = 42) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all approaches and generate comparison.
        
        Args:
            trained: Dictionary with trained models/agents
            workload_type: Type of workload
            n_steps: Number of steps per episode
            n_episodes: Number of evaluation episodes
            seed: Random seed
        
        Returns:
            Comparison results dictionary
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING ALL APPROACHES - {workload_type.upper()} WORKLOAD")
        print(f"{'='*70}\n")
        
        # Create evaluation environment
        env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed+100, workload_type=workload_type)
        
        # Run comprehensive comparison
        results = self.comparator.compare_all_approaches(
            traditional_agents=trained["traditional"],
            vpq_agent=trained["vpq_inspired"],
            dqn_model=trained["dqn"],
            env=env,
            n_episodes=n_episodes
        )
        
        return results
    
    def generate_reports(self, results: Dict[str, Dict[str, Any]], workload_type: str) -> Dict[str, str]:
        """
        Generate all reports and visualizations.
        
        Args:
            results: Comparison results
            workload_type: Type of workload
        
        Returns:
            Dictionary mapping report types to file paths
        """
        print(f"\nGenerating reports for {workload_type} workload...")
        
        report_paths = {}
        workload_dir = os.path.join(self.output_dir, workload_type)
        os.makedirs(workload_dir, exist_ok=True)
        
        # Comparison table
        table_path = os.path.join(workload_dir, "comparison_table.csv")
        self.comparator.generate_comparison_table(results, save_path=table_path)
        report_paths["comparison_table"] = table_path
        
        # Comparison report
        report_path = os.path.join(workload_dir, "comparison_report.txt")
        self.comparator.generate_comparison_report(
            results, workload_type=workload_type, save_path=report_path
        )
        report_paths["comparison_report"] = report_path
        
        # JSON results
        json_path = os.path.join(workload_dir, "detailed_results.json")
        self.comparator.save_results(results, filename=os.path.basename(json_path))
        report_paths["detailed_results"] = json_path
        
        # Visualizations
        plot_paths = plot_comprehensive_comparison(
            results, output_dir=workload_dir, prefix=f"{workload_type}_comparison"
        )
        report_paths.update(plot_paths)
        
        return report_paths
    
    def demonstrate_adaptive_decisions(self, dqn_model: DQN, workload_type: str = "steady",
                                      n_steps: int = 300, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Demonstrate adaptive decision making with different scenarios.
        
        Args:
            dqn_model: Trained DQN model
            workload_type: Type of workload
            n_steps: Number of steps
            seed: Random seed
        
        Returns:
            List of decision dictionaries
        """
        print(f"\n{'='*70}")
        print(f"ADAPTIVE DECISION DEMONSTRATION - {workload_type.upper()} WORKLOAD")
        print(f"{'='*70}\n")
        
        env = EnhancedCloudCostGym(n_steps=n_steps, seed=seed, workload_type=workload_type)
        decision_maker = AdaptiveDecisionMaker(dqn_model, env)
        
        decisions = decision_maker.demonstrate_adaptive_behavior(n_scenarios=3)
        
        # Save decisions
        decisions_dir = os.path.join(self.output_dir, workload_type, "adaptive_decisions")
        os.makedirs(decisions_dir, exist_ok=True)
        
        decisions_path = os.path.join(decisions_dir, "decisions.json")
        with open(decisions_path, 'w') as f:
            # Convert to serializable format
            serializable_decisions = []
            for decision in decisions:
                serializable = {
                    "service_name": decision["service_name"],
                    "scale_action": decision["scale_action_name"],
                    "expected_cost": float(decision["expected_cost"]),
                    "scenario_type": decision["scenario_type"],
                    "reasoning": decision["reasoning"]
                }
                serializable_decisions.append(serializable)
            json.dump(serializable_decisions, f, indent=2)
        
        print(f"\nAdaptive decisions saved to {decisions_path}")
        
        return decisions
    
    def run_complete_evaluation(self, workload_types: List[str] = None,
                               n_steps: int = 300, n_episodes: int = 5,
                               seed: int = 42) -> Dict[str, Any]:
        """
        Run complete evaluation across all workload types.
        
        Args:
            workload_types: List of workload types to evaluate
            n_steps: Number of steps per episode
            n_episodes: Number of evaluation episodes
            seed: Random seed
        
        Returns:
            Complete evaluation results
        """
        if workload_types is None:
            workload_types = ["steady", "diurnal", "batch", "bursty"]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION - ALL WORKLOAD TYPES")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for workload_type in workload_types:
            print(f"\n{'#'*80}")
            print(f"# WORKLOAD TYPE: {workload_type.upper()}")
            print(f"{'#'*80}\n")
            
            # Train all approaches
            trained = self.train_all_approaches(
                workload_type=workload_type,
                n_steps=n_steps,
                seed=seed
            )
            
            # Evaluate all approaches
            results = self.evaluate_all_approaches(
                trained=trained,
                workload_type=workload_type,
                n_steps=n_steps,
                n_episodes=n_episodes,
                seed=seed
            )
            
            # Generate reports
            report_paths = self.generate_reports(results, workload_type)
            
            # Demonstrate adaptive decisions
            decisions = self.demonstrate_adaptive_decisions(
                dqn_model=trained["dqn"],
                workload_type=workload_type,
                n_steps=n_steps,
                seed=seed
            )
            
            all_results[workload_type] = {
                "results": results,
                "report_paths": report_paths,
                "decisions": decisions
            }
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results: Dict[str, Any]) -> None:
        """Generate summary report across all workload types."""
        summary_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE EVALUATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for workload_type, data in all_results.items():
                f.write(f"\n{workload_type.upper()} WORKLOAD:\n")
                f.write("-" * 80 + "\n")
                
                results = data["results"]
                
                # Find best performers
                best_cost = min(results.items(), key=lambda x: x[1].get("total_cost", float('inf')))
                best_sla = min(results.items(), key=lambda x: x[1].get("sla_violation_rate", float('inf')))
                
                f.write(f"Best Cost: {best_cost[0]} (${best_cost[1].get('total_cost', 0):.2f})\n")
                f.write(f"Best SLA: {best_sla[0]} ({best_sla[1].get('sla_violation_rate', 0):.2%})\n")
        
        print(f"\nSummary report saved to {summary_path}")


# Test the comprehensive evaluator
if __name__ == "__main__":
    evaluator = ComprehensiveEvaluator(output_dir="test_outputs")
    
    # Run evaluation on a single workload type for testing
    trained = evaluator.train_all_approaches(workload_type="steady", n_steps=100, seed=42)
    results = evaluator.evaluate_all_approaches(trained, workload_type="steady", 
                                               n_steps=100, n_episodes=2, seed=42)
    report_paths = evaluator.generate_reports(results, "steady")
    
    print("\nEvaluation completed!")
    print("Generated reports:")
    for name, path in report_paths.items():
        print(f"  {name}: {path}")

