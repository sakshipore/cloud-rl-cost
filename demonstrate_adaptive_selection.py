# demonstrate_adaptive_selection.py
"""
Demonstration Script for Adaptive Service Selection

This script demonstrates how the DQN agent adaptively selects different services
based on different scenarios and workload conditions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from rl.adaptive_decision import AdaptiveDecisionMaker
import os
import json
from typing import Dict, List


def create_scenario_environments() -> Dict[str, EnhancedCloudCostGym]:
    """
    Create different scenario environments to test adaptive selection.
    
    Returns:
        Dictionary of scenario names to environments
    """
    scenarios = {}
    
    # Scenario 1: Low demand, cost-sensitive
    scenarios["low_demand_cost_sensitive"] = EnhancedCloudCostGym(
        n_steps=50, seed=100, workload_type="steady"
    )
    # Modify workload to be low
    low_workload = np.random.randint(50, 150, 50)
    scenarios["low_demand_cost_sensitive"].env.workload = low_workload
    
    # Scenario 2: High demand, latency-critical
    scenarios["high_demand_latency_critical"] = EnhancedCloudCostGym(
        n_steps=50, seed=200, workload_type="steady"
    )
    # Modify workload to be high
    high_workload = np.random.randint(400, 600, 50)
    scenarios["high_demand_latency_critical"].env.workload = high_workload
    
    # Scenario 3: Variable demand, balanced
    scenarios["variable_demand_balanced"] = EnhancedCloudCostGym(
        n_steps=50, seed=300, workload_type="diurnal"
    )
    
    # Scenario 4: Bursty demand
    scenarios["bursty_demand"] = EnhancedCloudCostGym(
        n_steps=50, seed=400, workload_type="bursty"
    )
    
    return scenarios


def analyze_service_selection_pattern(model: DQN, env: EnhancedCloudCostGym, 
                                     scenario_name: str, n_steps: int = 50) -> Dict:
    """
    Analyze service selection patterns for a given scenario.
    
    Args:
        model: Trained DQN model
        env: Environment for the scenario
        scenario_name: Name of the scenario
        n_steps: Number of steps to analyze
    
    Returns:
        Dictionary with selection analysis
    """
    obs, _ = env.reset()
    
    service_selections = {
        "ec2_ondemand": 0,
        "ec2_spot": 0,
        "lambda": 0,
        "fargate": 0
    }
    
    scaling_actions = {
        "scale_down": 0,
        "no_change": 0,
        "scale_up": 0
    }
    
    service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
    scale_actions = ["scale_down", "no_change", "scale_up"]
    
    selection_history = []
    demand_history = []
    latency_history = []
    cost_history = []
    
    for step in range(n_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Decode action
        service_type = int(action) // 3
        scale_action = int(action) % 3
        
        selected_service = service_names[service_type]
        scale_action_name = scale_actions[scale_action]
        
        # Record selection
        service_selections[selected_service] += 1
        scaling_actions[scale_action_name] += 1
        selection_history.append({
            "step": step,
            "service": selected_service,
            "scale_action": scale_action_name,
            "demand": obs[0],
            "utilization": obs[1],
            "latency": obs[2]
        })
        
        demand_history.append(obs[0])
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        latency_history.append(info.get("current_latency", 0))
        if "service_costs" in info:
            cost_history.append(sum(info["service_costs"].values()))
        else:
            cost_history.append(0.0)
        
        if done:
            break
    
    return {
        "scenario": scenario_name,
        "service_selections": service_selections,
        "scaling_actions": scaling_actions,
        "selection_history": selection_history,
        "demand_history": demand_history,
        "latency_history": latency_history,
        "cost_history": cost_history,
        "total_steps": len(selection_history)
    }


def plot_service_selection_analysis(analyses: List[Dict], save_path: str = None):
    """
    Plot service selection patterns across different scenarios.
    
    Args:
        analyses: List of analysis dictionaries
        save_path: Path to save plot
    """
    n_scenarios = len(analyses)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.style.use("seaborn-v0_8-whitegrid")
    
    service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Service selection frequency
    ax1 = axes[0, 0]
    x_pos = np.arange(len(service_names))
    width = 0.8 / n_scenarios
    
    for i, analysis in enumerate(analyses):
        counts = [analysis["service_selections"][s] for s in service_names]
        ax1.bar(x_pos + i * width, counts, width, label=analysis["scenario"], 
               alpha=0.8, color=colors[i % len(colors)])
    
    ax1.set_xlabel("Service Type", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Selection Count", fontsize=12, fontweight='bold')
    ax1.set_title("Service Selection Frequency by Scenario", fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos + width * (n_scenarios - 1) / 2)
    ax1.set_xticklabels([s.replace("_", " ").title() for s in service_names], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Service selection over time for first scenario
    if analyses:
        ax2 = axes[0, 1]
        analysis = analyses[0]
        history = analysis["selection_history"]
        
        service_indices = {s: i for i, s in enumerate(service_names)}
        selection_sequence = [service_indices[h["service"]] for h in history]
        
        ax2.plot(selection_sequence, marker='o', markersize=4, linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Time Step", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Service Type", fontsize=12, fontweight='bold')
        ax2.set_title(f"Service Selection Over Time: {analysis['scenario']}", 
                     fontsize=14, fontweight='bold')
        ax2.set_yticks(range(len(service_names)))
        ax2.set_yticklabels([s.replace("_", " ").title() for s in service_names])
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Demand vs Service Selection
    ax3 = axes[1, 0]
    for analysis in analyses:
        history = analysis["selection_history"]
        demands = [h["demand"] for h in history]
        services = [h["service"] for h in history]
        
        service_colors_map = {s: colors[i] for i, s in enumerate(service_names)}
        for i, (demand, service) in enumerate(zip(demands, services)):
            ax3.scatter(i, demand, c=service_colors_map[service], 
                       s=50, alpha=0.6, label=service if i == 0 else "")
    
    ax3.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Demand (req/s)", fontsize=12, fontweight='bold')
    ax3.set_title("Demand Pattern with Service Selection", fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost and Latency over time
    ax4 = axes[1, 1]
    if analyses:
        analysis = analyses[0]
        time_steps = range(len(analysis["cost_history"]))
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(time_steps, analysis["cost_history"], 
                        color='#1f77b4', label='Cost', linewidth=2)
        line2 = ax4_twin.plot(time_steps, analysis["latency_history"], 
                             color='#ff7f0e', label='Latency', linewidth=2)
        
        ax4.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Latency Target')
        
        ax4.set_xlabel("Time Step", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Cost ($)", fontsize=12, fontweight='bold', color='#1f77b4')
        ax4_twin.set_ylabel("Latency (ms)", fontsize=12, fontweight='bold', color='#ff7f0e')
        ax4.set_title(f"Cost and Latency Over Time: {analysis['scenario']}", 
                     fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def demonstrate_adaptive_selection(model_path: str = None, output_dir: str = "research_outputs_final/adaptive_selection_demo"):
    """
    Demonstrate adaptive service selection across different scenarios.
    
    Args:
        model_path: Path to trained DQN model (if None, will train one)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ADAPTIVE SERVICE SELECTION DEMONSTRATION")
    print("=" * 80)
    
    # Load or train model
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"\nLoading trained model from {model_path}...")
        model = DQN.load(model_path)
    else:
        print("\nTraining DQN model for demonstration...")
        env = EnhancedCloudCostGym(n_steps=200, seed=42, workload_type="steady")
        model = DQN("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=20000, progress_bar=True)
        print("Model trained!")
    
    # Create scenario environments
    print("\nCreating scenario environments...")
    scenarios = create_scenario_environments()
    
    # Analyze each scenario
    print("\nAnalyzing service selection patterns...")
    analyses = []
    
    for scenario_name, env in scenarios.items():
        print(f"  Analyzing: {scenario_name}")
        analysis = analyze_service_selection_pattern(model, env, scenario_name, n_steps=50)
        analyses.append(analysis)
        
        # Print summary
        print(f"    Service selections: {analysis['service_selections']}")
        print(f"    Scaling actions: {analysis['scaling_actions']}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_path = os.path.join(output_dir, "adaptive_selection_analysis.png")
    plot_service_selection_analysis(analyses, save_path=plot_path)
    print(f"  Saved: {plot_path}")
    
    # Save detailed results
    results_path = os.path.join(output_dir, "selection_analysis.json")
    with open(results_path, 'w') as f:
        # Convert to serializable format
        serializable = []
        for analysis in analyses:
            serializable.append({
                "scenario": analysis["scenario"],
                "service_selections": analysis["service_selections"],
                "scaling_actions": analysis["scaling_actions"],
                "total_steps": analysis["total_steps"]
            })
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {results_path}")
    
    # Create decision maker and demonstrate reasoning
    print("\nDemonstrating adaptive decision reasoning...")
    decision_maker = AdaptiveDecisionMaker(model, scenarios["low_demand_cost_sensitive"])
    
    decisions = []
    for scenario_name in ["latency_critical", "cost_sensitive", "balanced"]:
        obs, _ = scenarios["low_demand_cost_sensitive"].reset(seed=42)
        decision = decision_maker.make_decision(obs, scenario_type=scenario_name, verbose=False)
        decisions.append(decision)
        
        print(f"\n{scenario_name.upper()} Scenario:")
        print(f"  Selected: {decision['service_name']}")
        print(f"  Action: {decision['scale_action_name']}")
        print(f"  Reasoning: {decision['reasoning'][:200]}...")
    
    # Save decisions
    decisions_path = os.path.join(output_dir, "adaptive_decisions.json")
    with open(decisions_path, 'w') as f:
        serializable_decisions = []
        for decision in decisions:
            serializable_decisions.append({
                "scenario": decision["scenario_type"],
                "service": decision["service_name"],
                "scale_action": decision["scale_action_name"],
                "expected_cost": float(decision["expected_cost"]),
                "reasoning": decision["reasoning"]
            })
        json.dump(serializable_decisions, f, indent=2)
    print(f"  Saved: {decisions_path}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey Findings:")
    for analysis in analyses:
        most_used = max(analysis["service_selections"].items(), key=lambda x: x[1])
        print(f"  {analysis['scenario']}: Most used service = {most_used[0]} ({most_used[1]} times)")


if __name__ == "__main__":
    # Try to use existing trained model
    model_path = "research_outputs_final/models/steady/dqn_model"
    
    if not os.path.exists(model_path + ".zip"):
        print("No trained model found. Training new model...")
        model_path = None
    
    demonstrate_adaptive_selection(model_path=model_path, output_dir="research_outputs_final/adaptive_selection_demo")

