# envs/enhanced_cloud_env.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from envs.services import CloudService, get_all_services
from envs.workloads import generate_workload

class EnhancedCloudEnvironment:
    """
    Enhanced cloud environment that simulates multiple service types with different
    pricing models, reliability characteristics, and performance profiles.
    """
    
    def __init__(self, n_steps: int = 1440, seed: Optional[int] = None, 
                 workload_type: str = "diurnal", services: Optional[Dict[str, CloudService]] = None):
        """
        Initialize the enhanced cloud environment.
        
        Args:
            n_steps: Number of simulation steps
            seed: Random seed for reproducibility
            workload_type: Type of workload pattern to generate
            services: Dictionary of available services (uses defaults if None)
        """
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
        self.workload_type = workload_type
        
        # Initialize services
        if services is None:
            self.services = get_all_services()
        else:
            self.services = services
        
        # Service state tracking
        self.service_instances = {name: 0 for name in self.services.keys()}
        self.service_pending = {name: [] for name in self.services.keys()}
        
        # SLA and performance parameters
        self.latency_target = 200  # ms
        self.sla_penalty = 5.0  # Increased from 2.0 to prioritize latency more
        self.availability_penalty = 1.5
        self.high_load_threshold = 0.8  # Utilization threshold for high load
        self.high_load_bonus = 0.5  # Bonus for meeting SLA under high load
        
        # Current state
        self.t = 0
        self.workload = None
        self.current_latency = 100.0
        
        # Metrics tracking
        self.history = {
            "demand": [],
            "latency": [],
            "total_cost": [],
            "sla_violations": [],
            "service_usage": {name: [] for name in self.services.keys()},
            "service_costs": {name: [] for name in self.services.keys()},
            "interruptions": [],
            "service_interruptions": {name: [] for name in self.services.keys()},
            "availability": {name: [] for name in self.services.keys()},
            "availability_violations": []
        }
        
        # Initialize workload
        self.workload = generate_workload(n_steps, seed, workload_type)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        # Start with at least 1 instance to handle initial demand (warm start)
        # This helps agents learn proper scaling behavior
        initial_instances = {name: 1 for name in self.services.keys()}
        self.service_instances = initial_instances
        self.service_pending = {name: [] for name in self.services.keys()}
        self.current_latency = 100.0
        
        # Generate new workload
        self.workload = generate_workload(self.n_steps, seed, self.workload_type)
        
        # Reset history
        self.history = {
            "demand": [],
            "latency": [],
            "total_cost": [],
            "sla_violations": [],
            "service_usage": {name: [] for name in self.services.keys()},
            "service_costs": {name: [] for name in self.services.keys()},
            "interruptions": [],
            "service_interruptions": {name: [] for name in self.services.keys()},
            "availability": {name: [] for name in self.services.keys()},
            "availability_violations": [],
            "reward_components": {
                "cost_term": [],
                "latency_penalty": [],
                "availability_penalty": [],
                "high_load_bonus": [],
                "zero_capacity_penalty": [],
                "total_reward": []
            }
        }
    
    def step(self, action: Tuple[int, int]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.
        
        Args:
            action: Tuple of (service_type, scale_action)
                   service_type: 0=EC2 On-Demand, 1=EC2 Spot, 2=Lambda, 3=Fargate
                   scale_action: 0=scale down, 1=no change, 2=scale up
        
        Returns:
            Tuple of (reward, done, info)
        """
        if self.t >= self.n_steps:
            return 0.0, True, {}
        
        service_type, scale_action = action
        service_names = list(self.services.keys())
        
        if service_type >= len(service_names):
            service_type = 0  # Default to first service
        
        selected_service = service_names[service_type]
        service = self.services[selected_service]
        
        # Handle scaling
        self._handle_scaling(selected_service, scale_action)
        
        # Process pending instances
        self._process_pending_instances()
        
        # Handle service interruptions (especially for spot instances)
        service_interruptions = self._handle_interruptions()
        total_interruptions = sum(service_interruptions.values())
        
        # Calculate current demand and capacity
        current_demand = self.workload[self.t]
        total_capacity = self._calculate_total_capacity()
        
        # Calculate latency based on utilization
        utilization = current_demand / total_capacity if total_capacity > 0 else 1.0
        latency = self._calculate_latency(utilization)
        self.current_latency = latency
        
        # Calculate costs for each service
        total_cost = 0.0
        service_costs = {}
        
        for name, service in self.services.items():
            instances = self.service_instances[name]
            if instances > 0:
                # Calculate service-specific demand
                service_demand = self._calculate_service_demand(name, current_demand)
                
                # Calculate cost
                cost = service.calculate_cost(
                    instances=instances,
                    requests=service_demand * 60,  # Convert to requests per minute
                    duration=1.0,  # 1 minute
                    time=self.t
                )
                service_costs[name] = cost
                total_cost += cost
            else:
                service_costs[name] = 0.0
        
        # Calculate availability for each service
        # Availability = 1 - (interruption_rate) for this step
        service_availability = {}
        for name, service in self.services.items():
            instances = self.service_instances[name]
            if instances > 0:
                # Calculate interruption rate for this service
                interruption_rate = service_interruptions[name] / instances if instances > 0 else 0.0
                availability = 1.0 - interruption_rate
            else:
                availability = 1.0  # No instances means no interruptions
            service_availability[name] = availability
        
        # Check overall availability violation (if any service has availability < target)
        overall_availability_violation = 0
        for name, service in self.services.items():
            if hasattr(service, 'availability_target'):
                if service_availability[name] < service.availability_target:
                    overall_availability_violation = 1
                    break
        
        # Calculate SLA violation (latency)
        sla_violation = 1 if latency > self.latency_target else 0
        
        # Calculate reward using enhanced reward function
        reward, reward_components = self._calculate_reward(
            total_cost, latency, utilization, service_availability, 
            overall_availability_violation, total_capacity, current_demand
        )
        
        # Update history
        self._update_history(current_demand, latency, total_cost, sla_violation, 
                           service_costs, total_interruptions, service_interruptions,
                           service_availability, overall_availability_violation,
                           reward_components)
        
        # Advance time
        self.t += 1
        done = self.t >= self.n_steps
        
        # Prepare info
        info = {
            "service_used": selected_service,
            "sla_violation": sla_violation,
            "availability_violation": overall_availability_violation,
            "interrupted": total_interruptions,
            "service_interruptions": service_interruptions,
            "service_availability": service_availability,
            "utilization": utilization,
            "service_costs": service_costs
        }
        
        return reward, done, info
    
    def _handle_scaling(self, service_name: str, scale_action: int) -> None:
        """Handle scaling actions for a specific service."""
        if scale_action == 0:  # Scale down
            self.service_instances[service_name] = max(0, self.service_instances[service_name] - 1)
        elif scale_action == 2:  # Scale up
            service = self.services[service_name]
            self.service_pending[service_name].append(service.startup_time)
    
    def _process_pending_instances(self) -> None:
        """Process pending instances and activate them when ready."""
        for service_name in self.services.keys():
            # Decrease pending time
            self.service_pending[service_name] = [
                time - 1 for time in self.service_pending[service_name]
            ]
            
            # Activate ready instances
            ready_instances = [
                time for time in self.service_pending[service_name] if time <= 0
            ]
            self.service_instances[service_name] += len(ready_instances)
            
            # Remove activated instances from pending
            self.service_pending[service_name] = [
                time for time in self.service_pending[service_name] if time > 0
            ]
    
    def _handle_interruptions(self) -> Dict[str, int]:
        """Handle service interruptions (mainly for spot instances).
        
        Returns:
            Dictionary mapping service names to number of interruptions
        """
        service_interruptions = {name: 0 for name in self.services.keys()}
        total_interruptions = 0
        
        for service_name, service in self.services.items():
            instances = self.service_instances[service_name]
            if instances > 0 and service.reliability < 1.0:
                # Check for interruptions
                for _ in range(instances):
                    if self.rng.random() > service.reliability:
                        service_interruptions[service_name] += 1
                        total_interruptions += 1
                        self.service_instances[service_name] -= 1
        
        return service_interruptions
    
    def _calculate_total_capacity(self) -> float:
        """Calculate total capacity across all services."""
        total_capacity = 0.0
        for service_name, service in self.services.items():
            instances = self.service_instances[service_name]
            total_capacity += instances * service.capacity
        return total_capacity
    
    def _calculate_service_demand(self, service_name: str, total_demand: int) -> int:
        """Calculate demand allocated to a specific service."""
        # Simple proportional allocation based on capacity
        service_capacity = self.service_instances[service_name] * self.services[service_name].capacity
        total_capacity = self._calculate_total_capacity()
        
        if total_capacity == 0:
            return 0
        
        return int(total_demand * (service_capacity / total_capacity))
    
    def _calculate_latency(self, utilization: float) -> float:
        """Calculate latency based on utilization."""
        if utilization <= 0.6:
            return 120.0
        elif utilization <= 0.8:
            return 120.0 + (utilization - 0.6) * 300.0
        else:
            return 180.0 + (utilization - 0.8) * 1000.0
    
    def _calculate_reward(self, total_cost: float, latency: float, utilization: float,
                         service_availability: Dict[str, float],
                         availability_violation: int, total_capacity: float = 0.0,
                         current_demand: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward function balancing cost minimization and SLA compliance.
        
        This reward function is a key contribution of the research, designed to balance:
        1. Cost minimization (primary objective)
        2. SLA compliance (latency and availability constraints)
        3. Performance stability (incentive for meeting SLA under high load)
        
        Components:
        1. Cost term: -total_cost (negative because we minimize cost)
        2. SLA latency penalty: -sla_penalty if latency > target
        3. SLA availability penalty: -availability_penalty if availability < target
        4. High-load incentive: +high_load_bonus if SLA met under high load (utilization > threshold)
        
        Args:
            total_cost: Total cost for this time step
            latency: Current latency in milliseconds
            utilization: Current resource utilization (0-1)
            service_availability: Dictionary mapping service names to availability values
            availability_violation: 1 if availability violation occurred, 0 otherwise
            total_capacity: Total capacity across all services
            current_demand: Current workload demand
        
        Returns:
            Tuple of (total_reward, reward_components_dict)
            - total_reward: Combined reward value (higher is better)
            - reward_components: Dictionary with individual reward components for analysis
        """
        # Component 1: Cost term (negative because we want to minimize cost)
        cost_term = -total_cost
        
        # Component 2: SLA latency penalty (progressive - worse latency = bigger penalty)
        sla_violation = 1 if latency > self.latency_target else 0
        if sla_violation:
            # Progressive penalty: penalty increases with how much latency exceeds target
            latency_excess = latency - self.latency_target
            # Base penalty + progressive component (more penalty for worse violations)
            latency_penalty = -self.sla_penalty - (latency_excess / 100.0)  # Additional 0.01 per ms over target
        else:
            latency_penalty = 0.0
        
        # Component 3: SLA availability penalty
        availability_penalty_value = -self.availability_penalty * availability_violation
        
        # Component 4: High-load incentive
        # Reward for meeting SLA under high load conditions
        high_load_bonus = 0.0
        if utilization >= self.high_load_threshold:
            # Under high load, if we meet both latency and availability targets, give bonus
            if sla_violation == 0 and availability_violation == 0:
                high_load_bonus = self.high_load_bonus
        
        # Component 5: Zero capacity penalty (prevent agents from using 0 instances)
        # This is critical to prevent agents from learning "no instances = no cost"
        zero_capacity_penalty = 0.0
        if total_capacity == 0 and current_demand > 0:
            # Heavy penalty for having no capacity when there's demand
            # Make it proportional to demand to ensure it's always worse than cost
            zero_capacity_penalty = -50.0 - (current_demand * 0.1)  # Large penalty to discourage this behavior
        
        # Total reward (higher is better)
        total_reward = cost_term + latency_penalty + availability_penalty_value + high_load_bonus + zero_capacity_penalty
        
        # Store components for analysis
        reward_components = {
            "cost_term": cost_term,
            "latency_penalty": latency_penalty,
            "availability_penalty": availability_penalty_value,
            "high_load_bonus": high_load_bonus,
            "zero_capacity_penalty": zero_capacity_penalty,
            "total_reward": total_reward
        }
        
        return total_reward, reward_components
    
    def _update_history(self, demand: int, latency: float, total_cost: float, 
                       sla_violation: int, service_costs: Dict[str, float], 
                       interruptions: int, service_interruptions: Dict[str, int],
                       service_availability: Dict[str, float],
                       availability_violation: int,
                       reward_components: Dict[str, float]) -> None:
        """Update the history with current step data."""
        self.history["demand"].append(demand)
        self.history["latency"].append(latency)
        self.history["total_cost"].append(total_cost)
        self.history["sla_violations"].append(sla_violation)
        self.history["interruptions"].append(interruptions)
        self.history["availability_violations"].append(availability_violation)
        
        # Store reward components
        self.history["reward_components"]["cost_term"].append(reward_components["cost_term"])
        self.history["reward_components"]["latency_penalty"].append(reward_components["latency_penalty"])
        self.history["reward_components"]["availability_penalty"].append(reward_components["availability_penalty"])
        self.history["reward_components"]["high_load_bonus"].append(reward_components["high_load_bonus"])
        self.history["reward_components"]["total_reward"].append(reward_components["total_reward"])
        
        for service_name in self.services.keys():
            self.history["service_usage"][service_name].append(self.service_instances[service_name])
            self.history["service_costs"][service_name].append(service_costs.get(service_name, 0.0))
            self.history["service_interruptions"][service_name].append(
                service_interruptions.get(service_name, 0)
            )
            self.history["availability"][service_name].append(
                service_availability.get(service_name, 1.0)
            )
    
    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        current_demand = self.workload[self.t] if self.t < len(self.workload) else 0
        total_capacity = self._calculate_total_capacity()
        utilization = current_demand / total_capacity if total_capacity > 0 else 1.0
        
        # Get current prices for each service
        prices = []
        for service_name, service in self.services.items():
            price = service.calculate_cost(1, 0, 1.0, self.t)  # Price for 1 instance, 1 minute
            prices.append(price)
        
        # State vector: [demand, utilization, latency, service_instances..., prices...]
        state = [current_demand, utilization, self.current_latency]
        state.extend([self.service_instances[name] for name in self.services.keys()])
        state.extend(prices)
        
        return np.array(state, dtype=np.float32)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the current episode."""
        if not self.history["demand"]:
            return {}
        
        total_cost = sum(self.history["total_cost"])
        sla_violations = sum(self.history["sla_violations"])
        total_interruptions = sum(self.history["interruptions"])
        availability_violations = sum(self.history["availability_violations"])
        total_steps = len(self.history["demand"])
        
        # Service usage statistics and availability
        service_usage = {}
        service_availability_metrics = {}
        
        for service_name in self.services.keys():
            usage = self.history["service_usage"][service_name]
            interruptions = self.history["service_interruptions"][service_name]
            availability_history = self.history["availability"][service_name]
            
            # Calculate availability metrics
            total_service_instances = sum(usage)
            total_service_interruptions = sum(interruptions)
            
            if total_service_instances > 0:
                # Overall availability = 1 - (total_interruptions / total_instances)
                interruption_rate = total_service_interruptions / total_service_instances
                overall_availability = 1.0 - interruption_rate
            else:
                overall_availability = 1.0  # No instances means perfect availability
            
            # Average availability over time
            avg_availability = np.mean(availability_history) if availability_history else 1.0
            
            service_usage[service_name] = {
                "avg_instances": np.mean(usage),
                "max_instances": np.max(usage),
                "total_cost": sum(self.history["service_costs"][service_name]),
                "total_interruptions": total_service_interruptions
            }
            
            service_availability_metrics[service_name] = {
                "overall_availability": overall_availability,
                "avg_availability": avg_availability,
                "min_availability": np.min(availability_history) if availability_history else 1.0,
                "interruption_rate": interruption_rate if total_service_instances > 0 else 0.0
            }
            
            # Check if service meets availability target
            service = self.services[service_name]
            if hasattr(service, 'availability_target'):
                service_availability_metrics[service_name]["meets_target"] = (
                    overall_availability >= service.availability_target
                )
                service_availability_metrics[service_name]["availability_target"] = service.availability_target
            else:
                service_availability_metrics[service_name]["meets_target"] = True
                service_availability_metrics[service_name]["availability_target"] = 1.0
        
        # Calculate overall availability (weighted by service usage)
        overall_availability = 0.0
        total_weight = 0.0
        for service_name in self.services.keys():
            weight = sum(self.history["service_usage"][service_name])
            if weight > 0:
                overall_availability += service_availability_metrics[service_name]["overall_availability"] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_availability = overall_availability / total_weight
        else:
            overall_availability = 1.0
        
        return {
            "total_cost": total_cost,
            "sla_violations": sla_violations,
            "sla_violation_rate": sla_violations / total_steps if total_steps > 0 else 0.0,
            "availability_violations": availability_violations,
            "availability_violation_rate": availability_violations / total_steps if total_steps > 0 else 0.0,
            "total_interruptions": total_interruptions,
            "overall_availability": overall_availability,
            "avg_latency": np.mean(self.history["latency"]),
            "max_latency": np.max(self.history["latency"]),
            "service_usage": service_usage,
            "service_availability": service_availability_metrics,
            "total_steps": total_steps
        }


# Test the enhanced environment
if __name__ == "__main__":
    # Test with different workload types
    for workload_type in ["steady", "batch", "diurnal", "bursty"]:
        print(f"\nTesting {workload_type} workload:")
        env = EnhancedCloudEnvironment(n_steps=100, seed=42, workload_type=workload_type)
        env.reset()
        
        # Simple test policy
        for _ in range(100):
            # Random action: (service_type, scale_action)
            service_type = np.random.randint(0, 4)
            scale_action = np.random.randint(0, 3)
            action = (service_type, scale_action)
            
            reward, done, info = env.step(action)
            if done:
                break
        
        # Print metrics
        metrics = env.get_metrics()
        print(f"  Total cost: ${metrics['total_cost']:.2f}")
        print(f"  SLA violations: {metrics['sla_violations']}")
        print(f"  Interruptions: {metrics['total_interruptions']}")
        print(f"  Avg latency: {metrics['avg_latency']:.1f}ms")
