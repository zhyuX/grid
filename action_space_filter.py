import numpy as np
import pandas as pd
import warnings
from random import randint

import os
import grid2op
from grid2op import make
import random
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNReward
from grid2op.Agent import BaseAgent, AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Reward import CloseToOverflowReward, CombinedReward, CombinedScaledReward, L2RPNReward, LinesReconnectedReward

class FilterAgent(AgentWithConverter):
    def __init__(self, env, observation_space, action_space, args=None):
        super(FilterAgent, self).__init__(action_space, action_space_converter=IdToAct)
        self.env = env
        self.obs_space = observation_space
        # print('Filtering actions..')
        # self.action_space.filter_action(self._filter_action)
        # print('Done')
        self.obs_size = self.obs_space.size()
        self.action_size = self.action_space.size()
        print('obs space:{}; action space: {}'.format(self.obs_size, self.action_size))
        self.all_actions = self.get_all_actions()
        self.counts = np.zeros(len(self.all_actions))
        print("get selected all actions: {}".format(len(self.all_actions)))

    def get_all_actions(self):
        all_actions_bus = self.action_space.get_all_unitary_topologies_set(self.action_space)
        all_actions_line = self.action_space.get_all_unitary_line_change(self.action_space)
        print("set bus actions: {}, change line actions: {}".format(len(all_actions_bus), len(all_actions_line)))
        return all_actions_bus + all_actions_line

    def _filter_action(self, action):
        MAX_ELEM = 10
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
        # return obs_vec[6:]
        return obs_vec[np.r_[1, 3, 6:self.obs_size]]

    def convert_act(self, encoded_act):
        """
        将my_act返回的动作编号转换回env能处理的字典形式
        """
        return super().convert_act(encoded_act)

    def my_act(self, obs, reward=None, done=False):
        """
        根据obs返回encoded action
        此时的action已取过item()，即单个int类型数据
        """
        best_score = -99999
        best_action = self.env.action_space({})
        best_act_id = -1
        for act_id, action in enumerate(self.all_actions):
            if self.env.action_space._is_legal(action, self.env):
                simulated_obs, simulated_reward, done, _ = obs.simulate(action)
                if not done:
                    score = self.get_socre(simulated_obs.rho.max(), simulated_reward)
                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_act_id = act_id

        return best_action, best_act_id

    def get_socre(self, max_rho, reward):
        print("reward: {} max_rho:{} score:{}".format(reward, max_rho, reward - max_rho))
        return reward - max_rho

    def reconnect_best_line(self, obs):
        best_action = self.action_space({})  # default do nothing
        disconnected_lines = np.where(obs.line_status == False)[0]
        reconnect = []
        for line in disconnected_lines:
            if obs.time_before_cooldown_line[line] == 0:
                reconnect.append(line)

        if len(reconnect) > 0:
            # best_line = reconnect[0]
            best_score = -999
            for line in reconnect:
                reconnect_action_action = env.action_space({"set_line_status": [(line, +1)]})
                obs_, reward, done, _ = obs.simulate(reconnect_action_action)  # 模拟连接线路
                score = self.get_socre(obs_.rho.max(), reward)
                if not done and score > best_score:
                    best_action = reconnect_action_action
                    best_score = score

        return best_action

if __name__ == '__main__':
    # 指定服务器GPU编号
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # create an environment
    env_name = "l2rpn_neurips_2020_track1_small"
    backend = LightSimBackend()
    env = grid2op.make(env_name, reward_class=CombinedReward, backend=backend)
    print('env creating:', env_name)
    env.seed(123)
    # set combined reward
    cr = env.get_reward_instance()
    cr.set_range(reward_min=-2.0, reward_max=2.0)
    cr.addReward("L2RPN", L2RPNReward(), 0.1)
    # cr.addReward("LinesReconnected", LinesReconnectedReward(), 1.0)
    cr.addReward("CloseToOverflow", CloseToOverflowReward(), 2.0)
    cr.initialize(env)
    my_agent = FilterAgent(env, env.observation_space, env.action_space)

    do_nothing_action = my_agent.action_space({})

    # 模拟过程
    CHRONICS_NUM = int(100000)
    chronics = np.random.randint(0, 2000000, CHRONICS_NUM)
    for env_id in chronics:
        print("env id:", env_id)
        env.set_id(env_id)
        env.reset()
        skip = random.randint(1, 500)
        env.fast_forward_chronics(skip)  # 随机快进到某个时间步

        obs = env.get_obs()
        for step in range(2500):
            if obs.rho.max() > 0.9:
                best_action, act_id = my_agent.my_act(obs)
                obs, reward, done, _ = env.step(best_action)
                # print("{} do topo action, rho {}".format(step, obs.rho.max()))
                if act_id >= 0:
                    my_agent.counts[act_id] += 1
            else:
                obs, reward, done, _ = env.step(do_nothing_action)
                # print("{} do nothing".format(step))
            if done:
                break
    np.save("./actions.npy", my_agent.counts)