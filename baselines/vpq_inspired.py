# baselines/vpq_inspired.py
"""
VpQ-inspired Tabular Q-Learning Baseline Implementation

This module implements a simplified tabular Q-learning agent inspired by VpQ-learning
for cloud cost optimization. Unlike the DQN approach, this baseline:
- Uses discrete state space (state discretization)
- Optimizes cost only (no SLA awareness)
- Uses standard Q-learning updates with monotonicity-inspired value updates
- Serves as a comparison baseline representing existing work in the literature
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
import pickle
import os


class VpQInspiredAgent:
    """
    VpQ-inspired tabular Q-learning agent for cost-only optimization.
    
    This agent implements a simplified version of VpQ-learning concepts:
    - Discrete state space through binning
    - Cost-only reward function (no SLA penalties)
    - Standard Q-learning with ε-greedy exploration
    - Monotonicity-inspired value updates (conceptual VpQ inspiration)
    """
    
    def __init__(self, n_services: int = 4, n_scale_actions: int = 3,
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01):
        """
        Initialize the VpQ-inspired agent.
        
        Args:
            n_services: Number of cloud services (default: 4)
            n_scale_actions: Number of scaling actions (default: 3: down, no change, up)
            learning_rate: Q-learning learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            min_epsilon: Minimum exploration rate
        """
        self.n_services = n_services
        self.n_scale_actions = n_scale_actions
        self.n_actions = n_services * n_scale_actions
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state -> action -> Q-value
        # State is represented as a tuple of discrete bins
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # State discretization parameters
        # Demand bins: [0-100, 100-200, 200-300, 300-400, 400+]
        self.demand_bins = [0, 100, 200, 300, 400, float('inf')]
        
        # Utilization bins: [0-0.3, 0.3-0.6, 0.6-0.8, 0.8-1.0]
        self.utilization_bins = [0.0, 0.3, 0.6, 0.8, 1.0]
        
        # Price bins: Low, Medium, High (based on percentiles)
        # We'll determine these dynamically based on observed prices
        self.price_bins = None
        self.price_percentiles = [33.33, 66.67]  # Percentiles for low/medium/high
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_updates": 0,
            "exploration_rate": []
        }
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretize continuous state into discrete bins.
        
        State vector: [demand, utilization, latency, service_instances..., prices...]
        
        Args:
            state: Continuous state vector
        
        Returns:
            Tuple of discrete state indices
        """
        demand = state[0]
        utilization = state[1]
        
        # Discretize demand
        demand_bin = 0
        for i, threshold in enumerate(self.demand_bins[1:], 1):
            if demand < threshold:
                demand_bin = i - 1
                break
        else:
            demand_bin = len(self.demand_bins) - 2
        
        # Discretize utilization
        utilization_bin = 0
        for i, threshold in enumerate(self.utilization_bins[1:], 1):
            if utilization < threshold:
                utilization_bin = i - 1
                break
        else:
            utilization_bin = len(self.utilization_bins) - 2
        
        # Discretize prices (if price bins are initialized)
        price_bins = []
        if self.price_bins is not None:
            n_services = len(self.price_bins)
            price_start_idx = 3 + n_services  # After demand, utilization, latency, instances
            
            for i in range(n_services):
                if price_start_idx + i < len(state):
                    price = state[price_start_idx + i]
                    price_bin = 0
                    for j, threshold in enumerate(self.price_bins[i][1:], 1):
                        if price < threshold:
                            price_bin = j - 1
                            break
                    else:
                        price_bin = len(self.price_bins[i]) - 2
                    price_bins.append(price_bin)
                else:
                    price_bins.append(0)
        else:
            # If price bins not initialized, use default (will be initialized during training)
            n_services = self.n_services
            price_bins = [0] * n_services
        
        # Combine all discrete components
        discrete_state = (demand_bin, utilization_bin) + tuple(price_bins)
        
        return discrete_state
    
    def initialize_price_bins(self, price_samples: np.ndarray) -> None:
        """
        Initialize price bins based on observed price samples.
        
        Args:
            price_samples: Array of price samples (n_samples, n_services)
        """
        n_services = price_samples.shape[1] if len(price_samples.shape) > 1 else 1
        
        if len(price_samples.shape) == 1:
            price_samples = price_samples.reshape(-1, 1)
        
        self.price_bins = []
        for i in range(n_services):
            prices = price_samples[:, i]
            p33 = np.percentile(prices, self.price_percentiles[0])
            p67 = np.percentile(prices, self.price_percentiles[1])
            self.price_bins.append([0, p33, p67, float('inf')])
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
        
        Returns:
            Action index
        """
        discrete_state = self.discretize_state(state)
        state_key = discrete_state
        
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Exploitation: best action according to Q-table
        best_action = 0
        best_value = self.Q[state_key][best_action]
        
        for action in range(self.n_actions):
            q_value = self.Q[state_key][action]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """
        Update Q-table using Q-learning update rule.
        
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
        
        With monotonicity-inspired constraint: ensure Q-values are non-decreasing
        during training (conceptual VpQ inspiration).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        state_key = discrete_state
        next_state_key = discrete_next_state
        
        # Current Q-value
        current_q = self.Q[state_key][action]
        
        # Next state value
        if done:
            next_value = 0.0
        else:
            # Find max Q-value for next state
            next_value = max([self.Q[next_state_key][a] for a in range(self.n_actions)], 
                            default=0.0)
        
        # Q-learning update
        target = reward + self.discount_factor * next_value
        new_q = current_q + self.learning_rate * (target - current_q)
        
        # Monotonicity-inspired update: ensure Q-values don't decrease too much
        # This is a simplified conceptual implementation inspired by VpQ monotonicity
        # In practice, we allow some decrease but track it
        if new_q < current_q * 0.5:  # Prevent drastic decreases
            new_q = current_q * 0.5
        
        self.Q[state_key][action] = new_q
        self.training_stats["total_updates"] += 1
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.training_stats["exploration_rate"].append(self.epsilon)
    
    def predict(self, state: np.ndarray) -> int:
        """
        Predict action (exploitation only, no exploration).
        
        Args:
            state: Current state
        
        Returns:
            Action index
        """
        return self.get_action(state, training=False)
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train the agent for one episode.
        
        Args:
            env: Environment to train on
            max_steps: Maximum steps per episode
        
        Returns:
            Episode statistics
        """
        state, _ = env.reset()
        
        # Initialize price bins if not done
        if self.price_bins is None:
            # Collect some price samples
            price_samples = []
            for _ in range(10):
                sample_state = env.get_state() if hasattr(env, 'get_state') else state
                if len(sample_state) > 3 + self.n_services:
                    prices = sample_state[3 + self.n_services:3 + 2 * self.n_services]
                    price_samples.append(prices)
                state, _, _, _, _ = env.step(env.action_space.sample())
            
            if price_samples:
                self.initialize_price_bins(np.array(price_samples))
            
            state, _ = env.reset()
        
        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Get action
            action = self.get_action(state, training=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract cost-only reward (VpQ-inspired: cost only, no SLA)
            # The environment gives reward = -(cost + SLA_penalty)
            # For VpQ-inspired, we only care about cost
            # So we extract cost from info or use a modified reward
            if "service_costs" in info:
                step_cost = sum(info["service_costs"].values())
            else:
                # Estimate cost from reward (if reward includes SLA penalty, this is approximate)
                step_cost = -reward  # Rough estimate
            
            # Cost-only reward (negative because we minimize cost)
            cost_only_reward = -step_cost
            total_cost += step_cost
            
            # Update Q-table
            self.update(state, action, cost_only_reward, next_state, done)
            
            total_reward += cost_only_reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Decay epsilon
        self.decay_epsilon()
        self.training_stats["episodes"] += 1
        
        return {
            "total_reward": total_reward,
            "total_cost": total_cost,
            "steps": steps,
            "epsilon": self.epsilon
        }
    
    def train(self, env, n_episodes: int = 100, max_steps_per_episode: int = 1000) -> Dict[str, Any]:
        """
        Train the agent for multiple episodes.
        
        Args:
            env: Environment to train on
            n_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
        
        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_costs = []
        episode_steps = []
        
        print(f"Training VpQ-inspired agent for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            episode_stats = self.train_episode(env, max_steps_per_episode)
            
            episode_rewards.append(episode_stats["total_reward"])
            episode_costs.append(episode_stats["total_cost"])
            episode_steps.append(episode_stats["steps"])
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_cost = np.mean(episode_costs[-10:])
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Avg Reward: {avg_reward:.2f}, Avg Cost: {avg_cost:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return {
            "episode_rewards": episode_rewards,
            "episode_costs": episode_costs,
            "episode_steps": episode_steps,
            "training_stats": self.training_stats
        }
    
    def evaluate(self, env, n_episodes: int = 5, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
        
        Returns:
            Evaluation statistics
        """
        episode_rewards = []
        episode_costs = []
        episode_steps = []
        sla_violations = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_violations = 0
            steps = 0
            
            for step in range(max_steps):
                action = self.predict(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if "service_costs" in info:
                    step_cost = sum(info["service_costs"].values())
                else:
                    step_cost = -reward
                
                episode_cost += step_cost
                episode_reward += reward
                
                if info.get("sla_violation", 0) > 0:
                    episode_violations += 1
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_steps.append(steps)
            sla_violations.append(episode_violations)
        
        return {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "avg_cost": np.mean(episode_costs),
            "std_cost": np.std(episode_costs),
            "avg_steps": np.mean(episode_steps),
            "avg_sla_violations": np.mean(sla_violations),
            "sla_violation_rate": np.mean(sla_violations) / np.mean(episode_steps) if np.mean(episode_steps) > 0 else 0.0
        }
    
    def save(self, filepath: str) -> None:
        """Save the agent to a file."""
        data = {
            "Q": dict(self.Q),
            "training_stats": self.training_stats,
            "epsilon": self.epsilon,
            "price_bins": self.price_bins,
            "n_services": self.n_services,
            "n_scale_actions": self.n_scale_actions
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.Q = defaultdict(lambda: defaultdict(float), data["Q"])
        self.training_stats = data["training_stats"]
        self.epsilon = data["epsilon"]
        self.price_bins = data["price_bins"]
        self.n_services = data["n_services"]
        self.n_scale_actions = data["n_scale_actions"]


# Test the VpQ-inspired agent
if __name__ == "__main__":
    from envs.enhanced_cloud_gym import EnhancedCloudCostGym
    
    # Create environment
    env = EnhancedCloudCostGym(n_steps=100, seed=42, workload_type="steady")
    
    # Create agent
    agent = VpQInspiredAgent(n_services=4, n_scale_actions=3)
    
    # Train agent
    training_stats = agent.train(env, n_episodes=50, max_steps_per_episode=100)
    
    # Evaluate agent
    eval_stats = agent.evaluate(env, n_episodes=5, max_steps=100)
    
    print("\nEvaluation Results:")
    print(f"Average Cost: ${eval_stats['avg_cost']:.2f}")
    print(f"Average Reward: {eval_stats['avg_reward']:.2f}")
    print(f"SLA Violation Rate: {eval_stats['sla_violation_rate']:.2%}")
    
    # Save agent
    os.makedirs("outputs", exist_ok=True)
    agent.save("outputs/vpq_inspired_agent.pkl")
    print("\nAgent saved to outputs/vpq_inspired_agent.pkl")

