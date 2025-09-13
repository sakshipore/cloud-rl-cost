# baselines/rule_based.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

class RuleBasedAgent(ABC):
    """
    Abstract base class for rule-based agents.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> int:
        """
        Predict action based on observation.
        
        Args:
            observation: Current state observation
            info: Additional environment information
            
        Returns:
            Combined action (service_type * 3 + scale_action)
        """
        pass
    
    def get_strategy_description(self) -> str:
        """Get a description of the strategy."""
        return f"Rule-based strategy: {self.name}"


class CostOptimizedAgent(RuleBasedAgent):
    """
    Cost-optimized rule-based agent that prioritizes the cheapest available service.
    """
    
    def __init__(self):
        super().__init__("Cost Optimized")
        self.service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
    
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Always choose the cheapest service that can handle the load.
        """
        # Extract observation components
        demand = observation[0]
        utilization = observation[1]
        latency = observation[2]
        
        # Get service instances and prices
        n_services = len(self.service_names)
        service_instances = observation[3:3+n_services]
        service_prices = observation[3+n_services:3+2*n_services]
        
        # Find the cheapest service
        cheapest_service = np.argmin(service_prices)
        
        # Determine scaling action based on utilization
        if utilization > 0.8:  # High utilization, scale up
            scale_action = 2
        elif utilization < 0.3 and service_instances[cheapest_service] > 0:  # Low utilization, scale down
            scale_action = 0
        else:  # Maintain current level
            scale_action = 1
        
        return cheapest_service * 3 + scale_action


class ReliabilityOptimizedAgent(RuleBasedAgent):
    """
    Reliability-optimized rule-based agent that prioritizes the most reliable service.
    """
    
    def __init__(self):
        super().__init__("Reliability Optimized")
        self.service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
        # Reliability scores (higher is better)
        self.reliability_scores = [0.999, 0.95, 0.999, 0.999]  # EC2 On-Demand, Spot, Lambda, Fargate
    
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Always choose the most reliable service (prefer on-demand over spot).
        """
        # Extract observation components
        demand = observation[0]
        utilization = observation[1]
        latency = observation[2]
        
        # Get service instances
        n_services = len(self.service_names)
        service_instances = observation[3:3+n_services]
        
        # Find the most reliable service
        most_reliable_service = np.argmax(self.reliability_scores)
        
        # Determine scaling action
        if utilization > 0.8:  # High utilization, scale up
            scale_action = 2
        elif utilization < 0.3 and service_instances[most_reliable_service] > 0:  # Low utilization, scale down
            scale_action = 0
        else:  # Maintain current level
            scale_action = 1
        
        return most_reliable_service * 3 + scale_action


class HybridAgent(RuleBasedAgent):
    """
    Hybrid rule-based agent that uses different strategies based on conditions.
    """
    
    def __init__(self):
        super().__init__("Hybrid")
        self.service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
        self.reliability_scores = [0.999, 0.95, 0.999, 0.999]
    
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Use different strategies based on current conditions:
        - High utilization: Use reliable service (on-demand)
        - Low utilization: Use cheap service (spot)
        - Medium utilization: Use serverless (lambda)
        """
        # Extract observation components
        demand = observation[0]
        utilization = observation[1]
        latency = observation[2]
        
        # Get service instances and prices
        n_services = len(self.service_names)
        service_instances = observation[3:3+n_services]
        service_prices = observation[3+n_services:3+2*n_services]
        
        # Choose service based on utilization
        if utilization > 0.8:  # High utilization - use reliable service
            service_type = 0  # EC2 On-Demand
        elif utilization < 0.3:  # Low utilization - use cheap service
            service_type = 1  # EC2 Spot
        else:  # Medium utilization - use serverless
            service_type = 2  # Lambda
        
        # Determine scaling action
        if utilization > 0.8:  # High utilization, scale up
            scale_action = 2
        elif utilization < 0.3 and service_instances[service_type] > 0:  # Low utilization, scale down
            scale_action = 0
        else:  # Maintain current level
            scale_action = 1
        
        return service_type * 3 + scale_action


class WorkloadAwareAgent(RuleBasedAgent):
    """
    Workload-aware rule-based agent that adapts to different workload patterns.
    """
    
    def __init__(self):
        super().__init__("Workload Aware")
        self.service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
        self.demand_history = []
        self.history_length = 10
    
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Adapt strategy based on workload patterns.
        """
        # Extract observation components
        demand = observation[0]
        utilization = observation[1]
        latency = observation[2]
        
        # Update demand history
        self.demand_history.append(demand)
        if len(self.demand_history) > self.history_length:
            self.demand_history.pop(0)
        
        # Get service instances and prices
        n_services = len(self.service_names)
        service_instances = observation[3:3+n_services]
        service_prices = observation[3+n_services:3+2*n_services]
        
        # Analyze workload pattern
        if len(self.demand_history) >= 5:
            demand_std = np.std(self.demand_history)
            demand_mean = np.mean(self.demand_history)
            cv = demand_std / demand_mean if demand_mean > 0 else 0
            
            if cv > 0.5:  # High variability - use serverless
                service_type = 2  # Lambda
            elif demand_mean > 400:  # High average demand - use on-demand
                service_type = 0  # EC2 On-Demand
            else:  # Low/medium demand - use spot
                service_type = 1  # EC2 Spot
        else:
            # Default to hybrid strategy
            if utilization > 0.8:
                service_type = 0  # EC2 On-Demand
            elif utilization < 0.3:
                service_type = 1  # EC2 Spot
            else:
                service_type = 2  # Lambda
        
        # Determine scaling action
        if utilization > 0.8:  # High utilization, scale up
            scale_action = 2
        elif utilization < 0.3 and service_instances[service_type] > 0:  # Low utilization, scale down
            scale_action = 0
        else:  # Maintain current level
            scale_action = 1
        
        return service_type * 3 + scale_action


class ThresholdAgent(RuleBasedAgent):
    """
    Threshold-based agent that uses simple thresholds for decision making.
    """
    
    def __init__(self, demand_threshold: int = 300, utilization_threshold: float = 0.7):
        super().__init__("Threshold Based")
        self.demand_threshold = demand_threshold
        self.utilization_threshold = utilization_threshold
        self.service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
    
    def predict(self, observation: np.ndarray, info: Optional[Dict] = None) -> Tuple[int, int]:
        """
        Use simple thresholds for service selection and scaling.
        """
        # Extract observation components
        demand = observation[0]
        utilization = observation[1]
        latency = observation[2]
        
        # Get service instances
        n_services = len(self.service_names)
        service_instances = observation[3:3+n_services]
        
        # Service selection based on demand threshold
        if demand > self.demand_threshold:
            service_type = 0  # EC2 On-Demand for high demand
        else:
            service_type = 1  # EC2 Spot for low demand
        
        # Scaling based on utilization threshold
        if utilization > self.utilization_threshold:
            scale_action = 2  # Scale up
        elif utilization < 0.3 and service_instances[service_type] > 0:
            scale_action = 0  # Scale down
        else:
            scale_action = 1  # No change
        
        return service_type * 3 + scale_action


# Factory function to create agents
def create_agent(agent_type: str, **kwargs) -> RuleBasedAgent:
    """
    Create a rule-based agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        **kwargs: Additional parameters for the agent
        
    Returns:
        RuleBasedAgent instance
    """
    if agent_type == "cost_optimized":
        return CostOptimizedAgent()
    elif agent_type == "reliability_optimized":
        return ReliabilityOptimizedAgent()
    elif agent_type == "hybrid":
        return HybridAgent()
    elif agent_type == "workload_aware":
        return WorkloadAwareAgent()
    elif agent_type == "threshold":
        return ThresholdAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Test the rule-based agents
if __name__ == "__main__":
    # Test different agents
    agents = [
        CostOptimizedAgent(),
        ReliabilityOptimizedAgent(),
        HybridAgent(),
        WorkloadAwareAgent(),
        ThresholdAgent()
    ]
    
    # Create a sample observation
    # [demand, utilization, latency, service_instances..., prices...]
    sample_obs = np.array([
        250.0,  # demand
        0.6,    # utilization
        150.0,  # latency
        2.0,    # ec2_ondemand instances
        1.0,    # ec2_spot instances
        0.0,    # lambda instances
        1.0,    # fargate instances
        0.9,    # ec2_ondemand price
        0.3,    # ec2_spot price
        0.1,    # lambda price
        1.0     # fargate price
    ])
    
    print("Testing rule-based agents:")
    print(f"Sample observation: {sample_obs}")
    print()
    
    for agent in agents:
        action = agent.predict(sample_obs)
        service_names = ["ec2_ondemand", "ec2_spot", "lambda", "fargate"]
        scale_actions = ["scale_down", "no_change", "scale_up"]
        
        print(f"{agent.name}:")
        print(f"  Service: {service_names[action[0]]}")
        print(f"  Action: {scale_actions[action[1]]}")
        print(f"  Description: {agent.get_strategy_description()}")
        print()
