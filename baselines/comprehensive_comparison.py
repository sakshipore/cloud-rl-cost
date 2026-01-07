# baselines/comprehensive_comparison.py
"""
Comprehensive Comparison Framework

This module implements a three-way comparison framework for evaluating:
1. Traditional heuristic-based approaches (rule-based agents)
2. VpQ-inspired RL baseline (tabular Q-learning, cost-only)
3. DQN-based adaptive service selection (SLA-aware)

The framework generates comprehensive metrics, comparison tables, and visualizations
suitable for academic evaluation and research paper inclusion.
"""
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import DQN
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent, RuleBasedAgent
from baselines.vpq_inspired import VpQInspiredAgent


class ComprehensiveComparator:
    """
    Comprehensive comparison framework for all three approaches.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the comprehensive comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Comparison metrics to track
        self.metrics = [
            "total_cost",
            "avg_cost_per_step",
            "sla_violation_rate",
            "availability_violation_rate",
            "overall_availability",
            "avg_latency",
            "max_latency",
            "total_interruptions",
            "service_utilization_efficiency",
            "cost_per_request"
        ]
    
    def evaluate_traditional_heuristic(self, agent: RuleBasedAgent, env: EnhancedCloudCostGym,
                                     n_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate traditional heuristic-based approach.
        
        Args:
            agent: Rule-based agent
            env: Environment
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics dictionary
        """
        all_metrics = []
        total_requests = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            episode_requests = 0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_requests += info.get("current_demand", 0)
            
            metrics = env.get_metrics()
            if metrics:
                all_metrics.append(metrics)
                total_requests += episode_requests
        
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics, total_requests, "traditional")
    
    def evaluate_vpq_inspired(self, agent: VpQInspiredAgent, env: EnhancedCloudCostGym,
                              n_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate VpQ-inspired RL baseline.
        
        Args:
            agent: VpQ-inspired agent
            env: Environment
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics dictionary
        """
        eval_results = agent.evaluate(env, n_episodes=n_episodes)
        
        # Get additional metrics from environment
        all_metrics = []
        total_requests = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            episode_requests = 0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_requests += info.get("current_demand", 0)
            
            metrics = env.get_metrics()
            if metrics:
                all_metrics.append(metrics)
                total_requests += episode_requests
        
        # Combine agent evaluation with environment metrics
        aggregated = self._aggregate_metrics(all_metrics, total_requests, "vpq_inspired")
        aggregated.update({
            "avg_reward": eval_results["avg_reward"],
            "std_reward": eval_results["std_reward"]
        })
        
        return aggregated
    
    def evaluate_dqn(self, model: DQN, env: EnhancedCloudCostGym,
                     n_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate DQN-based adaptive approach.
        
        Args:
            model: Trained DQN model
            env: Environment
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics dictionary
        """
        all_metrics = []
        total_requests = 0
        total_reward = 0.0
        
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            episode_requests = 0
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_requests += info.get("current_demand", 0)
                episode_reward += reward
            
            metrics = env.get_metrics()
            if metrics:
                all_metrics.append(metrics)
                total_requests += episode_requests
                total_reward += episode_reward
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics, total_requests, "dqn")
        aggregated["avg_reward"] = total_reward / n_episodes
        
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, Any]],
                          total_requests: float, approach: str) -> Dict[str, Any]:
        """
        Aggregate metrics across episodes.
        
        Args:
            all_metrics: List of metric dictionaries
            total_requests: Total requests handled
            approach: Approach name
        
        Returns:
            Aggregated metrics dictionary
        """
        if not all_metrics:
            return {}
        
        aggregated = {
            "approach": approach,
            "n_episodes": len(all_metrics)
        }
        
        # Aggregate cost metrics
        total_costs = [m.get("total_cost", 0) for m in all_metrics]
        aggregated["total_cost"] = np.mean(total_costs)
        aggregated["std_cost"] = np.std(total_costs)
        aggregated["avg_cost_per_step"] = aggregated["total_cost"] / np.mean([m.get("total_steps", 1) for m in all_metrics])
        
        # Aggregate SLA metrics
        sla_rates = [m.get("sla_violation_rate", 0) for m in all_metrics]
        aggregated["sla_violation_rate"] = np.mean(sla_rates)
        aggregated["std_sla_violation_rate"] = np.std(sla_rates)
        
        availability_rates = [m.get("availability_violation_rate", 0) for m in all_metrics]
        aggregated["availability_violation_rate"] = np.mean(availability_rates)
        
        availabilities = [m.get("overall_availability", 1.0) for m in all_metrics]
        aggregated["overall_availability"] = np.mean(availabilities)
        aggregated["std_availability"] = np.std(availabilities)
        
        # Aggregate latency metrics
        avg_latencies = [m.get("avg_latency", 0) for m in all_metrics]
        aggregated["avg_latency"] = np.mean(avg_latencies)
        aggregated["std_latency"] = np.std(avg_latencies)
        
        max_latencies = [m.get("max_latency", 0) for m in all_metrics]
        aggregated["max_latency"] = np.mean(max_latencies)
        
        # Aggregate interruption metrics
        interruptions = [m.get("total_interruptions", 0) for m in all_metrics]
        aggregated["total_interruptions"] = np.mean(interruptions)
        
        # Calculate efficiency metrics
        if total_requests > 0:
            aggregated["cost_per_request"] = aggregated["total_cost"] / total_requests
        else:
            aggregated["cost_per_request"] = 0.0
        
        # Service utilization efficiency (simplified)
        # This would require more detailed service usage data
        aggregated["service_utilization_efficiency"] = 0.0  # Placeholder
        
        return aggregated
    
    def compare_all_approaches(self, traditional_agents: Dict[str, RuleBasedAgent],
                              vpq_agent: VpQInspiredAgent, dqn_model: DQN,
                              env: EnhancedCloudCostGym, n_episodes: int = 5) -> Dict[str, Any]:
        """
        Compare all three approaches comprehensively.
        
        Args:
            traditional_agents: Dictionary of traditional heuristic agents
            vpq_agent: VpQ-inspired agent
            dqn_model: Trained DQN model
            env: Environment
            n_episodes: Number of evaluation episodes
        
        Returns:
            Comprehensive comparison results
        """
        results = {}
        
        # Evaluate traditional heuristics and combine them into one entry
        print("Evaluating traditional heuristic approaches...")
        traditional_results = []
        for name, agent in traditional_agents.items():
            print(f"  - {name}")
            traditional_results.append(self.evaluate_traditional_heuristic(
                agent, env, n_episodes
            ))
        
        # Aggregate all traditional approaches into a single "traditional" entry
        if traditional_results:
            results["traditional"] = self._aggregate_traditional_approaches(traditional_results)
        
        # Evaluate VpQ-inspired
        print("Evaluating VpQ-inspired RL baseline...")
        results["vpq_inspired"] = self.evaluate_vpq_inspired(vpq_agent, env, n_episodes)
        
        # Evaluate DQN
        print("Evaluating DQN-based adaptive approach...")
        results["dqn"] = self.evaluate_dqn(dqn_model, env, n_episodes)
        
        return results
    
    def _aggregate_traditional_approaches(self, traditional_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics from all traditional approaches into a single entry.
        
        Args:
            traditional_results: List of evaluation results for each traditional approach
        
        Returns:
            Aggregated metrics dictionary
        """
        if not traditional_results:
            return {}
        
        aggregated = {
            "approach": "traditional",
            "n_episodes": traditional_results[0].get("n_episodes", 0)
        }
        
        # Average all metrics across traditional approaches
        metrics_to_aggregate = [
            "total_cost", "avg_cost_per_step", "sla_violation_rate",
            "availability_violation_rate", "overall_availability",
            "avg_latency", "max_latency", "total_interruptions",
            "cost_per_request", "service_utilization_efficiency"
        ]
        
        for metric in metrics_to_aggregate:
            values = [r.get(metric, 0) for r in traditional_results if metric in r]
            if values:
                aggregated[metric] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values)
        
        # Calculate standard deviation for cost
        cost_values = [r.get("total_cost", 0) for r in traditional_results]
        if cost_values:
            aggregated["std_cost"] = np.std(cost_values)
        
        return aggregated
    
    def generate_comparison_table(self, results: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comparison table from results.
        
        Args:
            results: Comparison results dictionary
            save_path: Path to save CSV (optional)
        
        Returns:
            Comparison DataFrame
        """
        # Prepare data for DataFrame
        rows = []
        for approach_name, metrics in results.items():
            # Skip individual traditional approaches if we have aggregated "traditional"
            if approach_name.startswith("traditional_") and "traditional" in results:
                continue
                
            row = {
                "Approach": approach_name,
                "Total Cost": metrics.get("total_cost", 0),
                "Avg Cost/Step": metrics.get("avg_cost_per_step", 0),
                "SLA Violation Rate": metrics.get("sla_violation_rate", 0),
                "Availability Violation Rate": metrics.get("availability_violation_rate", 0),
                "Overall Availability": metrics.get("overall_availability", 0),
                "Avg Latency (ms)": metrics.get("avg_latency", 0),
                "Max Latency (ms)": metrics.get("max_latency", 0),
                "Total Interruptions": metrics.get("total_interruptions", 0),
                "Cost per Request": metrics.get("cost_per_request", 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Comparison table saved to {save_path}")
        
        return df
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]],
                                  workload_type: str = "unknown",
                                  save_path: Optional[str] = None) -> str:
        """
        Generate text report from comparison results.
        
        Args:
            results: Comparison results dictionary
            workload_type: Type of workload tested
            save_path: Path to save report (optional)
        
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nWorkload Type: {workload_type}")
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n" + "=" * 80)
        
        # Summary statistics
        report_lines.append("\nSUMMARY STATISTICS")
        report_lines.append("-" * 80)
        
        for approach_name, metrics in results.items():
            report_lines.append(f"\n{approach_name.upper()}:")
            report_lines.append(f"  Total Cost: ${metrics.get('total_cost', 0):.2f}")
            report_lines.append(f"  SLA Violation Rate: {metrics.get('sla_violation_rate', 0):.2%}")
            report_lines.append(f"  Overall Availability: {metrics.get('overall_availability', 0):.2%}")
            report_lines.append(f"  Average Latency: {metrics.get('avg_latency', 0):.2f} ms")
            report_lines.append(f"  Cost per Request: ${metrics.get('cost_per_request', 0):.6f}")
        
        # Best performers
        report_lines.append("\n" + "=" * 80)
        report_lines.append("BEST PERFORMERS")
        report_lines.append("-" * 80)
        
        # Filter out individual traditional approaches for best performer analysis
        filtered_results = {
            k: v for k, v in results.items()
            if not (k.startswith("traditional_") and "traditional" in results)
        }
        
        # Lowest cost
        best_cost = min(filtered_results.items(), key=lambda x: x[1].get("total_cost", float('inf')))
        report_lines.append(f"\nLowest Cost: {best_cost[0]} (${best_cost[1].get('total_cost', 0):.2f})")
        
        # Lowest SLA violation rate
        best_sla = min(filtered_results.items(), key=lambda x: x[1].get("sla_violation_rate", float('inf')))
        report_lines.append(f"Lowest SLA Violation Rate: {best_sla[0]} ({best_sla[1].get('sla_violation_rate', 0):.2%})")
        
        # Highest availability
        best_avail = max(filtered_results.items(), key=lambda x: x[1].get("overall_availability", 0))
        report_lines.append(f"Highest Availability: {best_avail[0]} ({best_avail[1].get('overall_availability', 0):.2%})")
        
        # Lowest latency
        best_latency = min(filtered_results.items(), key=lambda x: x[1].get("avg_latency", float('inf')))
        report_lines.append(f"Lowest Average Latency: {best_latency[0]} ({best_latency[1].get('avg_latency', 0):.2f} ms)")
        
        report_lines.append("\n" + "=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Comparison report saved to {save_path}")
        
        return report
    
    def save_results(self, results: Dict[str, Dict[str, Any]], filename: str = "comparison_results.json") -> str:
        """
        Save comparison results to JSON file.
        
        Args:
            results: Comparison results dictionary
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath


# Test the comprehensive comparison framework
if __name__ == "__main__":
    from stable_baselines3 import DQN
    
    # Create environment
    env = EnhancedCloudCostGym(n_steps=100, seed=42, workload_type="steady")
    
    # Create traditional agents
    traditional_agents = {
        "cost_optimized": create_agent("cost_optimized"),
        "hybrid": create_agent("hybrid"),
        "reliability_optimized": create_agent("reliability_optimized")
    }
    
    # Create and train VpQ agent
    print("Training VpQ-inspired agent...")
    vpq_agent = VpQInspiredAgent()
    vpq_agent.train(env, n_episodes=20, max_steps_per_episode=100)
    
    # Create and train DQN model
    print("Training DQN model...")
    dqn_model = DQN("MlpPolicy", env, verbose=0)
    dqn_model.learn(total_timesteps=2000)
    
    # Create comparator
    comparator = ComprehensiveComparator(output_dir="test_outputs")
    
    # Run comparison
    results = comparator.compare_all_approaches(
        traditional_agents, vpq_agent, dqn_model, env, n_episodes=3
    )
    
    # Generate outputs
    comparator.generate_comparison_table(results, save_path="test_outputs/comparison_table.csv")
    comparator.generate_comparison_report(results, workload_type="steady",
                                        save_path="test_outputs/comparison_report.txt")
    comparator.save_results(results, "comparison_results.json")
    
    print("\nComparison completed!")

