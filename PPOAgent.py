import grid2op
import torch
import torch.optim as optim
import math
import random
import numpy as np
from collections import namedtuple
from grid2op.Agent import BaseAgent, AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Reward import GameplayReward, L2RPNReward
import os

from PPO import PPO

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class PPOAgent(AgentWithConverter):
    def __init__(self, env, observation_space, action_space, args=None):
        super(PPOAgent, self).__init__(action_space, action_space_converter=IdToAct)
        self.env = env
        self.obs_space = observation_space
        print('Filtering actions..')
        self.action_space.filter_action(self._filter_action)
        print('Done')
        self.obs_size = self.obs_space.size()
        self.action_size = self.action_space.size()
        print('obs space:{}; action space: {}'.format(self.obs_size, self.action_size))
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ppo = PPO(self.obs_size, self.action_size)

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def convert_obs(self, observation):
        """
        将observation转换成numpy vector的形式，便于处理
        """
        obs_vec = observation.to_vect()
        return obs_vec

    def convert_act(self, encoded_act):
        """
        将my_act返回的动作编号转换回env能处理的字典形式
        """
        return super().convert_act(encoded_act)

    def my_act(self, transformed_obs, reward=None, done=False, steps_done=0):
        """
        根据obs返回encoded action
        此时的action已取过item()，即单个int类型数据
        """
        action, action_prob = self.ppo.select_action(transformed_obs)

        return action, action_prob

    def store_transition(self, transition):
        self.ppo.store_transition(transition)


if __name__ == '__main__':
    # 指定服务器GPU编号
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # create an environment
    env_name = "l2rpn_neurips_2020_track1_small"  # for example, other environments might be usable
    print('env creating:', env_name)
    env = grid2op.make(env_name, reward_class=L2RPNReward)
    env.seed(123)
    my_agent = PPOAgent(env, env.observation_space, env.action_space)
    print('observation size:{} | action size:{}'.format(my_agent.obs_size, my_agent.action_size))
    for epoch in range(10000):
        obs = my_agent.convert_obs(env.reset())
        cnt = 0
        total_reward = 0
        while True:
            action, action_prob = my_agent.my_act(obs)
            obs_next, reward, done, info = env.step(my_agent.convert_act(action))
            obs_next = my_agent.convert_obs(obs_next)
            trans = Transition(obs, action, action_prob, reward, obs_next)
            my_agent.store_transition(trans)

            obs = obs_next
            cnt += 1
            total_reward += reward
            if done:
                if len(my_agent.ppo.buffer) >= my_agent.ppo.batch_size:
                    my_agent.ppo.update(epoch)
                my_agent.ppo.writer.add_scalar('Steptime/steptime', cnt, global_step=epoch)
                break
        print('epoch:{}; steps:{}; avg reward:{:.3f}'.format(epoch, cnt, float(total_reward/cnt)))
