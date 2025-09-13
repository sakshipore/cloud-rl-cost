# demo.py - Simple demonstration of the enhanced cloud RL system
from envs.enhanced_cloud_gym import EnhancedCloudCostGym
from baselines.rule_based import create_agent
import numpy as np

def run_demo():
    print('=' * 60)
    print('CLOUD RL COST OPTIMIZATION - DEMONSTRATION')
    print('=' * 60)
    print()
    
    # Test different workload types
    workload_types = ['steady', 'diurnal', 'batch', 'bursty']
    strategies = ['cost_optimized', 'hybrid', 'reliability_optimized']
    
    for workload_type in workload_types:
        print(f'Testing {workload_type.upper()} workload:')
        print('-' * 40)
        
        # Create environment
        env = EnhancedCloudCostGym(n_steps=50, seed=42, workload_type=workload_type)
        
        # Test each strategy
        for strategy_name in strategies:
            agent = create_agent(strategy_name)
            obs, _ = env.reset()
            
            total_reward = 0
            sla_violations = 0
            
            for _ in range(50):
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                sla_violations += info.get('sla_violation', 0)
                if terminated or truncated:
                    break
            
            cost = -total_reward  # Convert reward to cost
            sla_rate = sla_violations / 50 * 100
            
            print(f'  {strategy_name:20}: ${cost:6.2f} (SLA: {sla_rate:4.1f}%)')
        
        print()
    
    print('=' * 60)
    print('DEMONSTRATION COMPLETED')
    print('=' * 60)

if __name__ == "__main__":
    run_demo()
