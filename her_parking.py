import gym
import highway_env
import numpy as np

from stable_baselines3 import HER, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("parking-v0")

# # Create 4 artificial transitions per real transition
# n_sampled_goal = 4

# # SAC hyperparams:
# model = HER(
#     "MlpPolicy",
#     env,
#     SAC,
#     n_sampled_goal=n_sampled_goal,
#     goal_selection_strategy="future",
#     # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
#     # we have to manually specify the max number of steps per episode
#     max_episode_length=100,
#     verbose=1,
#     buffer_size=int(1e6),
#     learning_rate=1e-3,
#     gamma=0.95,
#     batch_size=256,
#     online_sampling=True,
#     policy_kwargs=dict(net_arch=[256, 256, 256]),
# )

# model.learn(int(2e5))
# model.save("her_sac_highway")

# # Load saved model
# # Because it needs access to `env.compute_reward()`
# # HER must be loaded with the env

# del model
model = HER.load("her_sac_highway", env=env)

obs = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs = env.reset()