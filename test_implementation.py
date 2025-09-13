# test_implementation.py
"""
Test script to verify the enhanced implementation works correctly.
"""

import os
import sys
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_services():
    """Test the services module."""
    print("Testing services module...")
    
    try:
        from envs.services import get_all_services, create_service
        
        # Test service creation
        services = get_all_services()
        print(f"  ✓ Created {len(services)} services")
        
        # Test individual service
        service = create_service("ec2_ondemand")
        cost = service.calculate_cost(instances=2, requests=1000, duration=60, time=0)
        print(f"  ✓ Service cost calculation: ${cost:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Services test failed: {e}")
        return False

def test_workloads():
    """Test the workloads module."""
    print("Testing workloads module...")
    
    try:
        from envs.workloads import generate_workload, generate_workloads, get_workload_characteristics
        
        # Test workload generation
        workload = generate_workload(n_steps=100, seed=42, workload_type="diurnal")
        print(f"  ✓ Generated diurnal workload: {len(workload)} steps")
        
        # Test multiple workloads
        workloads = generate_workloads(n_steps=100, types=["steady", "batch"], seed=42)
        print(f"  ✓ Generated {len(workloads)} workload types")
        
        # Test characteristics
        char = get_workload_characteristics(workload)
        print(f"  ✓ Workload characteristics: mean={char['mean_demand']:.1f}, cv={char['cv']:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Workloads test failed: {e}")
        return False

def test_enhanced_environment():
    """Test the enhanced environment."""
    print("Testing enhanced environment...")
    
    try:
        from envs.enhanced_cloud_env import EnhancedCloudEnvironment
        
        # Create environment
        env = EnhancedCloudEnvironment(n_steps=50, seed=42, workload_type="steady")
        print(f"  ✓ Created environment with {len(env.services)} services")
        
        # Test reset
        env.reset()
        print(f"  ✓ Environment reset successful")
        
        # Test step
        action = (0, 1)  # EC2 On-Demand, no change
        reward, done, info = env.step(action)
        print(f"  ✓ Step successful: reward={reward:.2f}, done={done}")
        
        # Test state
        state = env.get_state()
        print(f"  ✓ State vector length: {len(state)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Enhanced environment test failed: {e}")
        return False

def test_enhanced_gym():
    """Test the enhanced Gym wrapper."""
    print("Testing enhanced Gym wrapper...")
    
    try:
        from envs.enhanced_cloud_gym import EnhancedCloudCostGym
        
        # Create environment
        env = EnhancedCloudCostGym(n_steps=50, seed=42, workload_type="steady")
        print(f"  ✓ Created Gym environment")
        print(f"  ✓ Action space: {env.action_space}")
        print(f"  ✓ Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Reset successful: obs shape={obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step successful: reward={reward:.2f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Enhanced Gym test failed: {e}")
        return False

def test_rule_based_agents():
    """Test the rule-based agents."""
    print("Testing rule-based agents...")
    
    try:
        from baselines.rule_based import create_agent
        
        # Test agent creation
        agents = ["cost_optimized", "reliability_optimized", "hybrid", "workload_aware"]
        for agent_type in agents:
            agent = create_agent(agent_type)
            print(f"  ✓ Created {agent_type} agent")
        
        # Test prediction
        agent = create_agent("cost_optimized")
        sample_obs = np.array([250.0, 0.6, 150.0, 2.0, 1.0, 0.0, 1.0, 0.9, 0.3, 0.1, 1.0])
        action = agent.predict(sample_obs)
        print(f"  ✓ Agent prediction: {action}")
        
        return True
    except Exception as e:
        print(f"  ✗ Rule-based agents test failed: {e}")
        return False

def test_evaluation_framework():
    """Test the evaluation framework."""
    print("Testing evaluation framework...")
    
    try:
        from rl.evaluate import ComprehensiveEvaluator
        from baselines.rule_based import create_agent
        from envs.enhanced_cloud_gym import EnhancedCloudCostGym
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator("test_outputs")
        print(f"  ✓ Created evaluator")
        
        # Create test strategy and environment
        strategy = create_agent("cost_optimized")
        env = EnhancedCloudCostGym(n_steps=20, seed=42, workload_type="steady")
        
        # Test evaluation
        results = evaluator.evaluate_strategy(strategy, env, n_episodes=2)
        print(f"  ✓ Strategy evaluation: {len(results['episodes'])} episodes")
        
        return True
    except Exception as e:
        print(f"  ✗ Evaluation framework test failed: {e}")
        return False

def test_comparison_utilities():
    """Test the comparison utilities."""
    print("Testing comparison utilities...")
    
    try:
        from baselines.compare import compare_strategies
        from baselines.rule_based import create_agent
        
        # Create test strategies
        strategies = {
            "cost_optimized": create_agent("cost_optimized"),
            "hybrid": create_agent("hybrid")
        }
        
        # Test comparison
        results = compare_strategies(
            strategies=strategies,
            workload_types=["steady"],
            n_episodes=2
        )
        print(f"  ✓ Strategy comparison: {len(results)} workload types")
        
        return True
    except Exception as e:
        print(f"  ✗ Comparison utilities test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ENHANCED CLOUD RL COST OPTIMIZATION")
    print("=" * 60)
    
    tests = [
        test_services,
        test_workloads,
        test_enhanced_environment,
        test_enhanced_gym,
        test_rule_based_agents,
        test_evaluation_framework,
        test_comparison_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
