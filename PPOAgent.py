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
from lightsim2grid import LightSimBackend
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
        self.ppo = PPO(self.obs_size-4, self.action_size)

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()

        # if act_dict["force_line"]["reconnections"]["count"] > 0:
        #     return False
        # if act_dict["force_line"]["disconnections"]["count"] > 0:
        #     return False
        # if len(act_dict["topology"]["assigned_bus"]) + len(act_dict["topology"]["disconnect_bus"]) + len(act_dict["redispatch"]["generators"]) > 0:
        #     return False

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # create an environment
    env_name = "l2rpn_neurips_2020_track1_small"
    backend = LightSimBackend()
    env = grid2op.make(env_name, reward_class=L2RPNReward, backend=backend)
    print('env creating:', env_name)
    env.seed(123)
    my_agent = PPOAgent(env, env.observation_space, env.action_space)
    print('observation size:{} | action size:{}'.format(my_agent.obs_size, my_agent.action_size))
    step_list = []
    do_nothing_action = my_agent.action_space({})
    print('开始训练')
    result_file = 'grid_result_0114.txt'  # path of saved result
    for epoch in range(1, 1+10000):
        env.set_id(epoch)
        obs_env = env.reset()
        rho = obs_env.rho.max()  # get the max rho among all lines
        obs = my_agent.convert_obs(obs_env)
        cnt, do_nothing_cnt = 0, 0
        total_reward = 0
        while True:
            if rho <= 0.85 and epoch > 200:   # secure state, do nothing
                disconnected_lines = np.where(obs_env.line_status == False)[0]
                reconnect = []
                for line in disconnected_lines:
                    if obs_env.time_before_cooldown_line[line] == 0:
                        reconnect.append(line)
                if len(reconnect) > 0:
                    best_line = reconnect[0]
                    best_reward = -999
                    for line in reconnect:
                        reconnect_action_action = env.action_space({"set_line_status": [(line, +1)]})
                        obs_, reward_, done_, _ = obs_env.simulate(reconnect_action_action) # 模拟连接线路
                        if reward_ > best_reward:
                            best_line = line
                            best_reward = reward_
                    reconnect_action_action = env.action_space({"set_line_status": [(best_line, +1)]})
                    obs_next_env, reward, done, info = env.step(reconnect_action_action)  # 重连line
                else:
                    obs_next_env, reward, done, info = env.step(do_nothing_action)
                do_nothing_cnt += 1
            else:
                action, action_prob = my_agent.my_act(obs)
                obs_next_env, reward, done, info = env.step(my_agent.convert_act(action))
                obs_next = my_agent.convert_obs(obs_next_env)
                trans = Transition(obs, action, action_prob, reward, obs_next)
                my_agent.store_transition(trans)

                obs = obs_next
            rho = obs_next_env.rho.max()
            cnt += 1
            total_reward += reward
            if done:
                if len(my_agent.ppo.buffer) >= 2 * my_agent.ppo.batch_size:
                    my_agent.ppo.update(epoch)
                # my_agent.ppo.writer.add_scalar('Steptime/steptime', cnt, global_step=epoch)
                break
        # print('epoch:{}; steps:{} (do nothing {}); avg reward:{:.3f}'.format(epoch, cnt, do_nothing_cnt, float(total_reward/cnt)))
        my_agent.ppo.writer.add_scalar('steps', cnt, global_step=epoch)
        my_agent.ppo.writer.add_scalar('rewards', total_reward/cnt, global_step=epoch)
        with open(result_file, "a+") as file:
            file.write('epoch:{}; steps:{}; avg reward:{:.3f}'.format(epoch, cnt, float(total_reward/cnt)) + '\n')
        step_list.append(cnt)
        if epoch % 200 == 0:
            print('***** avg steps:', np.mean(step_list))
            with open(result_file, "a+") as file:
                file.write('-- train epoch: {}  avg steps: {}'.format(epoch, np.mean(step_list)) + '\n')
            step_list = []
        # 验证阶段
        if epoch % 1000 == 0:
            print('<进入验证阶段>')
            my_agent.ppo.change_mode('eval')
            eval_step_list = []
            for test_epoch in range(500):
                cnt, do_nothing_cnt = 0, 0
                total_reward = 0

                env.set_id(1000000 + test_epoch*2)
                obs_env = env.reset()
                rho = obs_env.rho.max()  # get the max rho among all lines
                obs = my_agent.convert_obs(obs_env)
                while True:
                    if rho <= 0.85:  # secure state, do nothing
                        disconnected_lines = np.where(obs_env.line_status == False)[0]
                        reconnect = []
                        for line in disconnected_lines:
                            if obs_env.time_before_cooldown_line[line] == 0:
                                reconnect.append(line)
                        if len(reconnect) > 0:
                            best_line = reconnect[0]
                            best_reward = -999
                            for line in reconnect:
                                reconnect_action_action = env.action_space({"set_line_status": [(line, +1)]})
                                obs_, reward_, done_, _ = obs_env.simulate(reconnect_action_action)  # 模拟连接线路
                                if reward_ > best_reward:
                                    best_line = line
                                    best_reward = reward_
                            reconnect_action_action = env.action_space({"set_line_status": [(best_line, +1)]})
                            obs_next_env, reward, done, info = env.step(reconnect_action_action)  # 重连line
                        else:
                            obs_next_env, reward, done, info = env.step(do_nothing_action)
                        do_nothing_cnt += 1
                    else:
                        action, action_prob = my_agent.my_act(obs)
                        obs_next_env, reward, done, info = env.step(my_agent.convert_act(action))
                        obs_next = my_agent.convert_obs(obs_next_env)

                        obs = obs_next
                    rho = obs_next_env.rho.max()
                    cnt += 1
                    total_reward += reward
                    if done:
                        break
                eval_step_list.append(cnt)
            print('****** eval avg steps: {} *******'.format(np.mean(eval_step_list)))
            with open(result_file, "a+") as file:
                file.write('-- {} eval avg steps: {} --'.format(epoch, np.mean(eval_step_list)) + '\n')
            my_agent.ppo.change_mode('train')
            print('<返回训练阶段>')
