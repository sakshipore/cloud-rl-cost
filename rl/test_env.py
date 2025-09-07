# rl/test_env.py
from envs.cloud_gym import CloudCostGym

if __name__ == "__main__":
    env = CloudCostGym(n_steps=100, seed=42)
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(100):
        action = env.action_space.sample()  # random action
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print("Finished episode, Total Reward =", total_reward)
