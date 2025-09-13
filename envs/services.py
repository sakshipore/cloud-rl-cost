# envs/services.py
import numpy as np
from typing import Callable, Dict, Any

class CloudService:
    """
    Represents a cloud service type with specific characteristics.
    """
    
    def __init__(self, name: str, pricing_model: Callable, capacity: int, 
                 startup_time: int, reliability: float = 1.0, 
                 max_instances: int = None, execution_limit: int = None):
        """
        Initialize a cloud service.
        
        Args:
            name: Service name (e.g., "EC2 On-Demand")
            pricing_model: Function that calculates price based on usage
            capacity: Request handling capacity per instance
            startup_time: Minutes required to start up
            reliability: Probability of not being interrupted (0-1)
            max_instances: Maximum number of instances allowed
            execution_limit: Maximum execution time in minutes (for serverless)
        """
        self.name = name
        self.pricing_model = pricing_model
        self.capacity = capacity
        self.startup_time = startup_time
        self.reliability = reliability
        self.max_instances = max_instances
        self.execution_limit = execution_limit
    
    def calculate_cost(self, instances: int, requests: int, duration: float, 
                      time: int, **kwargs) -> float:
        """
        Calculate the cost for using this service.
        
        Args:
            instances: Number of instances
            requests: Number of requests handled
            duration: Duration of usage in minutes
            time: Current time step
            **kwargs: Additional parameters for pricing model
        
        Returns:
            Total cost for this service
        """
        return self.pricing_model(instances, requests, duration, time, **kwargs)
    
    def can_handle_workload(self, demand: int, instances: int) -> bool:
        """
        Check if this service can handle the given workload.
        
        Args:
            demand: Current demand in requests/sec
            instances: Number of active instances
            
        Returns:
            True if service can handle the workload
        """
        if self.max_instances and instances > self.max_instances:
            return False
        
        capacity = instances * self.capacity
        return demand <= capacity
    
    def get_utilization(self, demand: int, instances: int) -> float:
        """
        Calculate utilization percentage.
        
        Args:
            demand: Current demand in requests/sec
            instances: Number of active instances
            
        Returns:
            Utilization as a percentage (0-1)
        """
        if instances == 0:
            return 1.0 if demand > 0 else 0.0
        
        capacity = instances * self.capacity
        return min(1.0, demand / capacity)


def on_demand_pricing(instances: int, requests: int, duration: float, 
                     time: int, base_price: float = 0.9, **kwargs) -> float:
    """
    On-demand pricing model with stable pricing and small variations.
    
    Args:
        instances: Number of instances
        requests: Number of requests (not used for on-demand)
        duration: Duration in minutes
        time: Current time step
        base_price: Base price per instance per minute
        
    Returns:
        Total cost
    """
    # Small price variation (±5%)
    price_variation = 0.95 + 0.1 * np.random.random()
    return instances * base_price * price_variation * duration


def spot_pricing(instances: int, requests: int, duration: float, 
                time: int, base_price: float = 0.9, **kwargs) -> float:
    """
    Spot pricing model with high variability and significant discounts.
    
    Args:
        instances: Number of instances
        requests: Number of requests (not used for spot)
        duration: Duration in minutes
        time: Current time step
        base_price: Base price per instance per minute
        
    Returns:
        Total cost
    """
    # High discount variability (30-90% discount)
    discount = 0.3 + 0.6 * np.random.beta(2, 5)
    spot_price = base_price * (1 - discount)
    return instances * spot_price * duration


def serverless_pricing(instances: int, requests: int, duration: float, 
                      time: int, base_price: float = 0.9, **kwargs) -> float:
    """
    Serverless pricing model (pay per request + compute time).
    
    Args:
        instances: Number of instances (not used for serverless)
        requests: Number of requests
        duration: Duration in minutes
        time: Current time step
        base_price: Base price per minute of compute time
        
    Returns:
        Total cost
    """
    # Pay per request + compute time
    request_cost = requests * 0.0000002  # $0.20 per 1M requests
    compute_cost = duration * base_price * 0.1  # Reduced compute cost for serverless
    return request_cost + compute_cost


def container_pricing(instances: int, requests: int, duration: float, 
                     time: int, base_price: float = 0.9, **kwargs) -> float:
    """
    Container pricing model (similar to on-demand but with different base price).
    
    Args:
        instances: Number of instances
        requests: Number of requests (not used for containers)
        duration: Duration in minutes
        time: Current time step
        base_price: Base price per instance per minute
        
    Returns:
        Total cost
    """
    # Container pricing is typically 10-20% more expensive than on-demand
    container_price = base_price * 1.15
    return instances * container_price * duration


# Predefined service configurations
SERVICE_CONFIGS = {
    "ec2_ondemand": {
        "name": "EC2 On-Demand",
        "pricing_model": on_demand_pricing,
        "capacity": 150,
        "startup_time": 3,
        "reliability": 0.999,
        "max_instances": 20
    },
    "ec2_spot": {
        "name": "EC2 Spot",
        "pricing_model": spot_pricing,
        "capacity": 150,
        "startup_time": 3,
        "reliability": 0.95,
        "max_instances": 20
    },
    "lambda": {
        "name": "AWS Lambda",
        "pricing_model": serverless_pricing,
        "capacity": 100,
        "startup_time": 0,
        "reliability": 0.999,
        "max_instances": None,
        "execution_limit": 15  # 15 minutes max execution time
    },
    "fargate": {
        "name": "AWS Fargate",
        "pricing_model": container_pricing,
        "capacity": 120,
        "startup_time": 1,
        "reliability": 0.999,
        "max_instances": 15
    }
}


def create_service(service_type: str, **kwargs) -> CloudService:
    """
    Create a cloud service instance from predefined configurations.
    
    Args:
        service_type: Type of service to create
        **kwargs: Additional parameters to override defaults
        
    Returns:
        CloudService instance
    """
    if service_type not in SERVICE_CONFIGS:
        raise ValueError(f"Unknown service type: {service_type}")
    
    config = SERVICE_CONFIGS[service_type].copy()
    config.update(kwargs)
    
    return CloudService(**config)


def get_all_services() -> Dict[str, CloudService]:
    """
    Get all predefined service types.
    
    Returns:
        Dictionary mapping service types to CloudService instances
    """
    return {name: create_service(name) for name in SERVICE_CONFIGS.keys()}


# Test the service implementations
if __name__ == "__main__":
    # Test service creation
    services = get_all_services()
    
    for name, service in services.items():
        print(f"\n{service.name}:")
        print(f"  Capacity: {service.capacity} req/s")
        print(f"  Startup time: {service.startup_time} min")
        print(f"  Reliability: {service.reliability}")
        print(f"  Max instances: {service.max_instances}")
        
        # Test pricing
        cost = service.calculate_cost(instances=2, requests=1000, duration=60, time=0)
        print(f"  Cost for 2 instances, 60 min: ${cost:.2f}")
        
        # Test workload handling
        can_handle = service.can_handle_workload(demand=200, instances=2)
        print(f"  Can handle 200 req/s with 2 instances: {can_handle}")
        
        utilization = service.get_utilization(demand=200, instances=2)
        print(f"  Utilization: {utilization:.2%}")
