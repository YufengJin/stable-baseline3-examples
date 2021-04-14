import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

model = DQN('MlpPolicy', 'LunarLander-v2', verbose=1, exploration_final_eps=0.1, target_update_interval=250)

eval_env = gym.make('LunarLander-v2')


# Train the agent
#model.learn(total_timesteps=int(1e5))
# Save the agent
#model.save("dqn_lunar")
#del model  # delete trained model to demonstrate loading
#
model = DQN.load("dqn_lunar")
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
#
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")