import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../panda-gym')) # the directory of 'panda-gym'
import gymnasium as gym

import panda_gym
import uf_gym
from collections import OrderedDict
from stable_baselines3 import DDPG, HerReplayBuffer

# target training environment
env = gym.make('XArm6Reach-v3', render_mode="rgb_array")

hyperparameters = OrderedDict([
    ('batch_size', 2048),
    ('buffer_size', 1000000),
    ('gamma', 0.95),
    ('learning_rate', 0.001),
    ('learning_starts', 100),
    ('policy', 'MultiInputPolicy'),
    ('policy_kwargs', dict(net_arch=[512, 512, 512], n_critics=2)),
    ('replay_buffer_class', HerReplayBuffer),
    ('replay_buffer_kwargs', dict( goal_selection_strategy='future', n_sampled_goal=4)),
    ('tau', 0.05),
])
model = DDPG(**hyperparameters, env=env, verbose=1, tensorboard_log='logs/ddpg-XArm6Reach-v3')

try:
    model.learn(500_000) # try learning for 500K steps
except KeyboardInterrupt:
        print("KeyboardInterrupt detected!")
        pass
# save the model after finish or interrupted by Ctrl-C
model.save('./model/ddpg-XArm6Reach-v3.pkl')

env.close()
