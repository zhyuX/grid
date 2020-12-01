import grid2op
import time
import os
from grid2op.Agent import RandomAgent, DoNothingAgent
from grid2op.Reward import GameplayReward, L2RPNReward

from agent import DQNAgent

# 指定服务器GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# create an environment
env_name = "l2rpn_neurips_2020_track1_small"  # for example, other environments might be usable
env = grid2op.make(env_name, reward_class=L2RPNReward)
env.seed(123)
# chunk_size = 1024
# env.chronics_handler.set_chunk_size(chunk_size)
print('环境 {} 初始化成功'.format(env_name))

my_agent = DQNAgent(env, env.observation_space, env.action_space)
print('observation size:{} | action size:{}'.format(my_agent.obs_space, my_agent.action_size))
my_agent.train()