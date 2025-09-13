# envs/workloads.py - generates synthetic workload patterns
import numpy as np
from typing import Dict, List, Optional

def generate_workload(n_steps=1440, seed=None, workload_type="diurnal"):
    """
    Generate synthetic workload for one day (per-minute samples).
    
    Args:
        n_steps: Number of time steps (minutes)
        seed: Random seed for reproducibility
        workload_type: Type of workload pattern to generate
        
    Returns:
        Integer requests-per-second values (length n_steps)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    
    if workload_type == "diurnal":
        return _generate_diurnal_workload(t, rng)
    elif workload_type == "steady":
        return _generate_steady_workload(t, rng)
    elif workload_type == "batch":
        return _generate_batch_workload(t, rng)
    elif workload_type == "bursty":
        return _generate_bursty_workload(t, rng)
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")


def _generate_diurnal_workload(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate diurnal workload with day/night patterns and spikes."""
    # Base day-night cycle (two peaks per day)
    base = 200 + 150 * np.sin(2 * np.pi * t / len(t) * 2)

    # Add a few random spikes
    spikes = np.zeros(len(t), dtype=float)
    for _ in range(5):  # 5 random spike events
        pos = int(rng.integers(0, len(t)))
        height = float(rng.integers(200, 600))
        width = int(rng.integers(8, 30))
        end = min(len(t), pos + width)
        # create a triangular-ish spike
        ramp = np.linspace(1.0, 0.0, end - pos)
        spikes[pos:end] += height * ramp

    # Additive Gaussian noise
    noise = rng.normal(0, 20, size=len(t))

    demand = np.clip(base + spikes + noise, 0, None)
    return demand.astype(int)


def _generate_steady_workload(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate steady workload with consistent load and small variations."""
    # Base steady load
    base = 300
    
    # Small periodic variations
    periodic = 50 * np.sin(2 * np.pi * t / 60)  # Hourly variation
    
    # Random noise
    noise = rng.normal(0, 15, size=len(t))
    
    demand = np.clip(base + periodic + noise, 0, None)
    return demand.astype(int)


def _generate_batch_workload(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate batch workload with large spikes followed by idle periods."""
    demand = np.zeros(len(t), dtype=int)
    
    # Create 3-5 batch jobs
    num_batches = rng.integers(3, 6)
    for _ in range(num_batches):
        # Random start time - ensure we have room for batch duration
        max_start = max(1, len(t) - 30)  # At least 30 minutes for batch
        start = rng.integers(0, max_start)
        # Random duration (30-120 minutes, but not longer than remaining time)
        max_duration = min(121, len(t) - start)
        duration = rng.integers(30, max_duration) if max_duration > 30 else 30
        end = min(len(t), start + duration)
        
        # High load during batch
        batch_load = rng.integers(400, 800)
        demand[start:end] = batch_load
    
    # Add some background noise
    noise = rng.normal(0, 10, size=len(t))
    demand = np.clip(demand + noise, 0, None)
    
    return demand.astype(int)


def _generate_bursty_workload(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate bursty workload with unpredictable short spikes."""
    # Base low load
    base = 100
    
    # Frequent short bursts
    bursts = np.zeros(len(t), dtype=float)
    num_bursts = rng.integers(20, 40)  # Many short bursts
    
    for _ in range(num_bursts):
        # Random burst location
        pos = rng.integers(0, len(t))
        # Short duration (2-10 minutes)
        duration = rng.integers(2, 11)
        end = min(len(t), pos + duration)
        
        # High burst load
        burst_height = rng.integers(300, 700)
        # Exponential decay
        decay = np.exp(-np.arange(end - pos) / (duration / 3))
        bursts[pos:end] += burst_height * decay
    
    # Add noise
    noise = rng.normal(0, 25, size=len(t))
    
    demand = np.clip(base + bursts + noise, 0, None)
    return demand.astype(int)


def generate_workloads(n_steps: int = 1440, types: Optional[List[str]] = None, 
                      seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate multiple workload patterns.
    
    Args:
        n_steps: Number of time steps
        types: List of workload types to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping workload types to demand arrays
    """
    if types is None:
        types = ["steady", "batch", "diurnal", "bursty"]
    
    workloads = {}
    for workload_type in types:
        workloads[workload_type] = generate_workload(n_steps, seed, workload_type)
    
    return workloads


def get_workload_characteristics(workload: np.ndarray) -> Dict[str, float]:
    """
    Analyze workload characteristics.
    
    Args:
        workload: Demand array
        
    Returns:
        Dictionary with workload characteristics
    """
    mean_demand = float(np.mean(workload))
    return {
        "mean_demand": mean_demand,
        "std_demand": float(np.std(workload)),
        "max_demand": float(np.max(workload)),
        "min_demand": float(np.min(workload)),
        "cv": float(np.std(workload) / mean_demand) if mean_demand > 0 else 0.0,  # Coefficient of variation
        "peak_to_mean": float(np.max(workload) / mean_demand) if mean_demand > 0 else 0.0,
        "zero_percentage": float(np.sum(workload == 0) / len(workload) * 100)
    }


# Quick test / visualization if run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    w = generate_workload(1440, seed=42)
    print("First 20 values:", w[:20])
    plt.figure(figsize=(10,3))
    plt.plot(w, linewidth=1)
    plt.title("Synthetic workload (req/s) — 1440 minutes")
    plt.xlabel("Minute")
    plt.ylabel("Requests/sec")
    plt.tight_layout()
    plt.show()
