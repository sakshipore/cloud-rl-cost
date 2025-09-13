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
        self.sla_penalty = 2.0
        
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
            "interruptions": []
        }
        
        # Initialize workload
        self.workload = generate_workload(n_steps, seed, workload_type)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.service_instances = {name: 0 for name in self.services.keys()}
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
            "interruptions": []
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
        interruptions = self._handle_interruptions()
        
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
        
        # Calculate reward (negative cost with SLA penalty)
        sla_violation = 1 if latency > self.latency_target else 0
        penalty = self.sla_penalty * sla_violation
        reward = -(total_cost + penalty)
        
        # Update history
        self._update_history(current_demand, latency, total_cost, sla_violation, 
                           service_costs, interruptions)
        
        # Advance time
        self.t += 1
        done = self.t >= self.n_steps
        
        # Prepare info
        info = {
            "service_used": selected_service,
            "sla_violation": sla_violation,
            "interrupted": interruptions,
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
    
    def _handle_interruptions(self) -> int:
        """Handle service interruptions (mainly for spot instances)."""
        interruptions = 0
        
        for service_name, service in self.services.items():
            instances = self.service_instances[service_name]
            if instances > 0 and service.reliability < 1.0:
                # Check for interruptions
                for _ in range(instances):
                    if self.rng.random() > service.reliability:
                        interruptions += 1
                        self.service_instances[service_name] -= 1
        
        return interruptions
    
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
    
    def _update_history(self, demand: int, latency: float, total_cost: float, 
                       sla_violation: int, service_costs: Dict[str, float], 
                       interruptions: int) -> None:
        """Update the history with current step data."""
        self.history["demand"].append(demand)
        self.history["latency"].append(latency)
        self.history["total_cost"].append(total_cost)
        self.history["sla_violations"].append(sla_violation)
        self.history["interruptions"].append(interruptions)
        
        for service_name in self.services.keys():
            self.history["service_usage"][service_name].append(self.service_instances[service_name])
            self.history["service_costs"][service_name].append(service_costs.get(service_name, 0.0))
    
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
        
        # Service usage statistics
        service_usage = {}
        for service_name in self.services.keys():
            usage = self.history["service_usage"][service_name]
            service_usage[service_name] = {
                "avg_instances": np.mean(usage),
                "max_instances": np.max(usage),
                "total_cost": sum(self.history["service_costs"][service_name])
            }
        
        return {
            "total_cost": total_cost,
            "sla_violations": sla_violations,
            "sla_violation_rate": sla_violations / len(self.history["demand"]),
            "total_interruptions": total_interruptions,
            "avg_latency": np.mean(self.history["latency"]),
            "max_latency": np.max(self.history["latency"]),
            "service_usage": service_usage,
            "total_steps": len(self.history["demand"])
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
