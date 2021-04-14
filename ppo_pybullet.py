import os
import gym
import pybullet_envs

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

# env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
# # Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                    clip_obs=10.)

# model = PPO('MlpPolicy', env)
# model.learn(total_timesteps=2000)

# # Don't forget to save the VecNormalize statistics when saving the agent
# log_dir = "./ppo/"
# model.save(log_dir + "ppo_halfcheetah")
# stats_path = os.path.join(log_dir, "vec_normalize.pkl")
# env.save(stats_path)

# # To demonstrate loading
# del model, env

# Load the saved statistics
log_dir = "./ppo/"
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

# Load the agent
model = PPO.load(log_dir + "ppo_halfcheetah", env=env)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
      break