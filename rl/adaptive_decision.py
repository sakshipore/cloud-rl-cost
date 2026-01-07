# rl/adaptive_decision.py
"""
Adaptive Service Selection Decision Module

This module provides an adaptive decision-making system that selects the best cloud service
for a given scenario with clear reasoning. It supports different scenario types:
- Latency-critical applications
- Cost-sensitive batch jobs
- Balanced workloads

The module demonstrates adaptive behavior by analyzing cost vs SLA trade-offs and
providing clear explanations for service selection decisions.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import DQN
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from envs.services import get_all_services


class AdaptiveDecisionMaker:
    """
    Adaptive service selection decision maker with reasoning capabilities.
    """
    
    def __init__(self, model: DQN, env: EnhancedCloudCostGym):
        """
        Initialize the adaptive decision maker.
        
        Args:
            model: Trained DQN model
            env: Environment for decision making
        """
        self.model = model
        self.env = env
        self.services = get_all_services()
        self.service_names = list(self.services.keys())
        
        # Service characteristics for reasoning
        self.service_characteristics = {
            "ec2_ondemand": {
                "name": "EC2 On-Demand",
                "reliability": 0.999,
                "cost_level": "high",
                "latency_profile": "low",
                "best_for": ["latency_critical", "balanced"]
            },
            "ec2_spot": {
                "name": "EC2 Spot",
                "reliability": 0.95,
                "cost_level": "low",
                "latency_profile": "low",
                "best_for": ["cost_sensitive"]
            },
            "lambda": {
                "name": "AWS Lambda",
                "reliability": 0.999,
                "cost_level": "medium",
                "latency_profile": "very_low",
                "best_for": ["latency_critical", "balanced"]
            },
            "fargate": {
                "name": "AWS Fargate",
                "reliability": 0.999,
                "cost_level": "medium_high",
                "latency_profile": "low",
                "best_for": ["balanced"]
            }
        }
    
    def analyze_state(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Analyze current state to extract key information.
        
        Args:
            state: Current environment state
        
        Returns:
            Dictionary with state analysis
        """
        demand = state[0]
        utilization = state[1]
        latency = state[2]
        
        # Extract service instances and prices
        n_services = len(self.service_names)
        service_instances = state[3:3+n_services]
        service_prices = state[3+n_services:3+2*n_services]
        
        # Calculate current capacity
        total_capacity = sum(
            service_instances[i] * self.services[name].capacity
            for i, name in enumerate(self.service_names)
        )
        
        # Determine workload intensity
        if demand < 100:
            intensity = "low"
        elif demand < 300:
            intensity = "medium"
        else:
            intensity = "high"
        
        # Determine utilization level
        if utilization < 0.3:
            util_level = "low"
        elif utilization < 0.7:
            util_level = "medium"
        else:
            util_level = "high"
        
        return {
            "demand": demand,
            "utilization": utilization,
            "latency": latency,
            "intensity": intensity,
            "util_level": util_level,
            "total_capacity": total_capacity,
            "service_instances": {name: service_instances[i] for i, name in enumerate(self.service_names)},
            "service_prices": {name: service_prices[i] for i, name in enumerate(self.service_names)}
        }
    
    def get_model_prediction(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Get model prediction for current state.
        
        Args:
            state: Current state
        
        Returns:
            Tuple of (action, Q-values for all actions)
        """
        # Get action from model
        action, _ = self.model.predict(state, deterministic=True)
        
        # Get Q-values (if available)
        # Note: This requires accessing the model's Q-network directly
        try:
            # Convert state to tensor and get Q-values
            import torch
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model.q_net(state_tensor).numpy()[0]
        except:
            # If Q-values not accessible, use action only
            q_values = None
        
        return int(action), q_values
    
    def make_decision(self, state: np.ndarray, scenario_type: str = "balanced",
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Make adaptive service selection decision with reasoning.
        
        Args:
            state: Current environment state
            scenario_type: Type of scenario ("latency_critical", "cost_sensitive", "balanced")
            verbose: Whether to print reasoning
        
        Returns:
            Dictionary with decision and reasoning
        """
        # Analyze state
        state_analysis = self.analyze_state(state)
        
        # Get model prediction
        action, q_values = self.get_model_prediction(state)
        
        # Decode action
        service_type = action // 3
        scale_action = action % 3
        
        selected_service_name = self.service_names[service_type]
        scale_actions = ["scale_down", "no_change", "scale_up"]
        scale_action_name = scale_actions[scale_action]
        
        # Get service characteristics
        service_char = self.service_characteristics[selected_service_name]
        
        # Calculate expected metrics
        current_price = state_analysis["service_prices"][selected_service_name]
        current_instances = state_analysis["service_instances"][selected_service_name]
        
        # Estimate expected cost (simplified)
        if scale_action == 2:  # Scale up
            expected_instances = current_instances + 1
        elif scale_action == 0:  # Scale down
            expected_instances = max(0, current_instances - 1)
        else:
            expected_instances = current_instances
        
        expected_cost = expected_instances * current_price
        
        # Estimate SLA compliance based on service characteristics
        expected_latency = "low" if service_char["latency_profile"] in ["low", "very_low"] else "medium"
        expected_availability = service_char["reliability"]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            state_analysis, selected_service_name, service_char,
            scenario_type, scale_action_name, expected_cost, expected_latency,
            expected_availability, q_values
        )
        
        decision = {
            "action": action,
            "service_type": service_type,
            "service_name": selected_service_name,
            "scale_action": scale_action,
            "scale_action_name": scale_action_name,
            "expected_cost": expected_cost,
            "expected_latency": expected_latency,
            "expected_availability": expected_availability,
            "reasoning": reasoning,
            "state_analysis": state_analysis,
            "scenario_type": scenario_type
        }
        
        if verbose:
            self._print_decision(decision)
        
        return decision
    
    def _generate_reasoning(self, state_analysis: Dict[str, Any],
                           service_name: str, service_char: Dict[str, Any],
                           scenario_type: str, scale_action: str,
                           expected_cost: float, expected_latency: str,
                           expected_availability: float,
                           q_values: Optional[np.ndarray]) -> str:
        """
        Generate reasoning explanation for the decision.
        
        Args:
            state_analysis: State analysis dictionary
            service_name: Selected service name
            service_char: Service characteristics
            scenario_type: Scenario type
            scale_action: Scaling action
            expected_cost: Expected cost
            expected_latency: Expected latency level
            expected_availability: Expected availability
            q_values: Q-values for all actions (optional)
        
        Returns:
            Reasoning string
        """
        reasoning_parts = []
        
        # Scenario-based reasoning
        reasoning_parts.append(f"Scenario: {scenario_type.replace('_', ' ').title()}")
        reasoning_parts.append(f"Current workload: {state_analysis['intensity']} intensity "
                              f"({state_analysis['demand']:.0f} req/s)")
        reasoning_parts.append(f"Current utilization: {state_analysis['util_level']} "
                              f"({state_analysis['utilization']:.1%})")
        
        # Service selection reasoning
        reasoning_parts.append(f"\nSelected Service: {service_char['name']}")
        reasoning_parts.append(f"  - Reliability: {service_char['reliability']:.1%}")
        reasoning_parts.append(f"  - Cost Level: {service_char['cost_level']}")
        reasoning_parts.append(f"  - Latency Profile: {service_char['latency_profile']}")
        
        # Cost vs SLA trade-off
        if scenario_type == "latency_critical":
            reasoning_parts.append("\nReasoning: Latency-critical scenario prioritizes low latency and high reliability.")
            reasoning_parts.append(f"  {service_char['name']} selected for its {service_char['latency_profile']} latency profile.")
        elif scenario_type == "cost_sensitive":
            reasoning_parts.append("\nReasoning: Cost-sensitive scenario prioritizes cost minimization.")
            reasoning_parts.append(f"  {service_char['name']} selected for its {service_char['cost_level']} cost level.")
        else:  # balanced
            reasoning_parts.append("\nReasoning: Balanced scenario considers both cost and SLA.")
            reasoning_parts.append(f"  {service_char['name']} selected as a balance between cost and performance.")
        
        # Scaling reasoning
        if scale_action == "scale_up":
            reasoning_parts.append(f"\nScaling Decision: Scale up")
            reasoning_parts.append(f"  - Current capacity may be insufficient for demand")
            reasoning_parts.append(f"  - Expected instances: {state_analysis['service_instances'][service_name]:.0f} → "
                                 f"{state_analysis['service_instances'][service_name] + 1:.0f}")
        elif scale_action == "scale_down":
            reasoning_parts.append(f"\nScaling Decision: Scale down")
            reasoning_parts.append(f"  - Current capacity exceeds demand")
            reasoning_parts.append(f"  - Expected instances: {state_analysis['service_instances'][service_name]:.0f} → "
                                 f"{max(0, state_analysis['service_instances'][service_name] - 1):.0f}")
        else:
            reasoning_parts.append(f"\nScaling Decision: No change")
            reasoning_parts.append(f"  - Current capacity matches demand")
        
        # Expected outcomes
        reasoning_parts.append(f"\nExpected Outcomes:")
        reasoning_parts.append(f"  - Expected Cost: ${expected_cost:.2f}")
        reasoning_parts.append(f"  - Expected Latency: {expected_latency}")
        reasoning_parts.append(f"  - Expected Availability: {expected_availability:.1%}")
        
        # Q-value reasoning (if available)
        if q_values is not None:
            best_q = np.max(q_values)
            selected_q = q_values[action] if 'action' in locals() else q_values[0]
            reasoning_parts.append(f"\nModel Confidence:")
            reasoning_parts.append(f"  - Selected action Q-value: {selected_q:.2f}")
            reasoning_parts.append(f"  - Best Q-value: {best_q:.2f}")
            if selected_q == best_q:
                reasoning_parts.append(f"  - Decision matches optimal action")
            else:
                reasoning_parts.append(f"  - Decision differs from optimal by {best_q - selected_q:.2f}")
        
        return "\n".join(reasoning_parts)
    
    def _print_decision(self, decision: Dict[str, Any]) -> None:
        """Print decision with formatting."""
        print("=" * 70)
        print("ADAPTIVE SERVICE SELECTION DECISION")
        print("=" * 70)
        print(f"\nScenario Type: {decision['scenario_type'].replace('_', ' ').title()}")
        print(f"\nSelected Service: {decision['service_name']}")
        print(f"Scaling Action: {decision['scale_action_name']}")
        print(f"\n{decision['reasoning']}")
        print("\n" + "=" * 70)
    
    def demonstrate_adaptive_behavior(self, n_scenarios: int = 3,
                                     scenario_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Demonstrate adaptive behavior across different scenarios.
        
        Args:
            n_scenarios: Number of scenarios to demonstrate
            scenario_types: List of scenario types (default: all types)
        
        Returns:
            List of decision dictionaries
        """
        if scenario_types is None:
            scenario_types = ["latency_critical", "cost_sensitive", "balanced"]
        
        decisions = []
        
        for i, scenario_type in enumerate(scenario_types[:n_scenarios]):
            print(f"\n{'='*70}")
            print(f"DEMONSTRATION {i+1}: {scenario_type.replace('_', ' ').title()} Scenario")
            print(f"{'='*70}\n")
            
            # Reset environment
            state, _ = self.env.reset(seed=42 + i)
            
            # Make decision
            decision = self.make_decision(state, scenario_type=scenario_type, verbose=True)
            decisions.append(decision)
        
        return decisions


# Convenience function for making adaptive decisions
def make_adaptive_decision(state: np.ndarray, model: DQN, env: EnhancedCloudCostGym,
                           scenario_type: str = "balanced",
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Make adaptive service selection decision with reasoning.
    
    Args:
        state: Current environment state
        model: Trained DQN model
        env: Environment
        scenario_type: Type of scenario ("latency_critical", "cost_sensitive", "balanced")
        verbose: Whether to print reasoning
    
    Returns:
        Dictionary with decision and reasoning
    """
    decision_maker = AdaptiveDecisionMaker(model, env)
    return decision_maker.make_decision(state, scenario_type=scenario_type, verbose=verbose)


# Test the adaptive decision module
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from envs.enhanced_cloud_gym import EnhancedCloudCostGym
    
    # Create environment
    env = EnhancedCloudCostGym(n_steps=100, seed=42, workload_type="steady")
    
    # Load or create a model (for testing, we'll create a simple one)
    # In practice, you would load a trained model
    print("Creating a test model (in practice, load a trained model)...")
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)  # Quick training for demo
    
    # Create decision maker
    decision_maker = AdaptiveDecisionMaker(model, env)
    
    # Demonstrate adaptive behavior
    decisions = decision_maker.demonstrate_adaptive_behavior(n_scenarios=3)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for i, decision in enumerate(decisions, 1):
        print(f"\nScenario {i} ({decision['scenario_type']}):")
        print(f"  Service: {decision['service_name']}")
        print(f"  Action: {decision['scale_action_name']}")
        print(f"  Expected Cost: ${decision['expected_cost']:.2f}")

