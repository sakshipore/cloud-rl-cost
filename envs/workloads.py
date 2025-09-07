# envs/workloads.py - generates synthetic workload (step 1)
import numpy as np

def generate_workload(n_steps=1440, seed=None):
    """
    Generate synthetic workload for one day (per-minute samples).
    Combines a day-night base, a few random spikes, and noise.
    Returns integer requests-per-second values (length n_steps).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Base day-night cycle (two peaks per day)
    base = 200 + 150 * np.sin(2 * np.pi * t / n_steps * 2)

    # Add a few random spikes
    spikes = np.zeros(n_steps, dtype=float)
    for _ in range(5):  # 5 random spike events
        pos = int(rng.integers(0, n_steps))
        height = float(rng.integers(200, 600))
        width = int(rng.integers(8, 30))
        end = min(n_steps, pos + width)
        # create a triangular-ish spike
        ramp = np.linspace(1.0, 0.0, end - pos)
        spikes[pos:end] += height * ramp

    # Additive Gaussian noise
    noise = rng.normal(0, 20, size=n_steps)

    demand = np.clip(base + spikes + noise, 0, None)
    return demand.astype(int)


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
