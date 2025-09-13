# envs/enhanced_cloud_gym.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
from envs.enhanced_cloud_env import EnhancedCloudEnvironment
from envs.services import get_all_services

class EnhancedCloudCostGym(gym.Env):
    """
    Enhanced Gym environment wrapper for cloud cost optimization with multiple service types.
    
    Observation space: [demand, utilization, latency, service_instances..., prices...]
    Action space: MultiDiscrete([4, 3]) - (service_type, scale_action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, n_steps: int = 1440, seed: Optional[int] = None, 
                 workload_type: str = "diurnal", services: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced cloud cost optimization environment.
        
        Args:
            n_steps: Number of simulation steps
            seed: Random seed for reproducibility
            workload_type: Type of workload pattern
            services: Available services (uses defaults if None)
        """
        super().__init__()
        
        self.n_steps = n_steps
        self.workload_type = workload_type
        
        # Initialize the underlying environment
        self.env = EnhancedCloudEnvironment(
            n_steps=n_steps, 
            seed=seed, 
            workload_type=workload_type,
            services=services
        )
        
        # Get service information
        self.service_names = list(self.env.services.keys())
        self.n_services = len(self.service_names)
        
        # Define action and observation spaces
        # Action: Combined action space (service_type * 3 + scale_action)
        # service_type: 0=EC2 On-Demand, 1=EC2 Spot, 2=Lambda, 3=Fargate
        # scale_action: 0=scale down, 1=no change, 2=scale up
        # Total actions: 4 services * 3 scale actions = 12 actions
        self.action_space = spaces.Discrete(self.n_services * 3)
        
        # Observation: [demand, utilization, latency, service_instances..., prices...]
        # Total observation size: 3 (base) + n_services (instances) + n_services (prices) = 3 + 2*n_services
        obs_size = 3 + 2 * self.n_services
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset(seed=seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        
        self.env.reset(seed=seed)
        
        observation = self.env.get_state()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            action: Combined action (service_type * 3 + scale_action)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            action = self.action_space.sample()
        
        # Convert combined action to (service_type, scale_action)
        service_type = action // 3
        scale_action = action % 3
        combined_action = (service_type, scale_action)
        
        # Take step in underlying environment
        reward, done, step_info = self.env.step(combined_action)
        
        # Get new observation
        observation = self.env.get_state()
        
        # Prepare info
        info = self._get_info()
        info.update(step_info)
        
        # Check if episode is done
        terminated = done
        truncated = False  # We don't use truncation in this environment
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        return {
            "step": self.env.t,
            "total_steps": self.env.n_steps,
            "workload_type": self.env.workload_type,
            "service_instances": self.env.service_instances.copy(),
            "current_demand": self.env.workload[self.env.t] if self.env.t < len(self.env.workload) else 0,
            "current_latency": self.env.current_latency,
            "total_capacity": self.env._calculate_total_capacity()
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment.
        
        Args:
            mode: Render mode
            
        Returns:
            Rendered output (if applicable)
        """
        if mode == "human":
            info = self._get_info()
            print(f"Step {info['step']}/{info['total_steps']}")
            print(f"Demand: {info['current_demand']} req/s")
            print(f"Latency: {info['current_latency']:.1f}ms")
            print(f"Total Capacity: {info['total_capacity']:.0f} req/s")
            print("Service Instances:")
            for service_name, instances in info['service_instances'].items():
                print(f"  {service_name}: {instances}")
            print("-" * 40)
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the current episode."""
        return self.env.get_metrics()
    
    def get_history(self) -> Dict[str, Any]:
        """Get the complete history of the current episode."""
        return self.env.history.copy()
    
    @property
    def service_instances(self):
        """Access to service instances from underlying environment."""
        return self.env.service_instances
    
    @property
    def history(self):
        """Access to history from underlying environment."""
        return self.env.history
    
    @property
    def services(self):
        """Access to services from underlying environment."""
        return self.env.services
    
    @property
    def latency_target(self):
        """Access to latency target from underlying environment."""
        return self.env.latency_target
    
    def set_workload_type(self, workload_type: str) -> None:
        """Change the workload type for the environment."""
        self.workload_type = workload_type
        self.env.workload_type = workload_type
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about available services."""
        service_info = {}
        for name, service in self.services.items():
            service_info[name] = {
                "name": service.name,
                "capacity": service.capacity,
                "startup_time": service.startup_time,
                "reliability": service.reliability,
                "max_instances": service.max_instances,
                "execution_limit": service.execution_limit
            }
        return service_info


# Test the enhanced Gym environment
if __name__ == "__main__":
    # Test with different workload types
    for workload_type in ["steady", "batch", "diurnal", "bursty"]:
        print(f"\nTesting {workload_type} workload:")
        
        env = EnhancedCloudCostGym(n_steps=50, seed=42, workload_type=workload_type)
        obs, info = env.reset()
        
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Initial observation: {obs}")
        print(f"Available services: {list(env.get_service_info().keys())}")
        
        # Run a few steps
        total_reward = 0.0
        for step in range(10):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step < 3:  # Show first few steps
                env.render()
            
            if terminated:
                break
        
        print(f"Total reward after 10 steps: {total_reward:.2f}")
        
        # Get final metrics
        metrics = env.get_metrics()
        if metrics:
            print(f"Final metrics: {metrics}")
        
        print("-" * 60)
