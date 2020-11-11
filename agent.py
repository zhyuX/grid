import grid2op
import torch
import torch.optim as optim
import math
import random
import numpy as np
from grid2op.Agent import BaseAgent, AgentWithConverter
from grid2op.Converter import IdToAct
import torch.nn.functional as F

from DQN import Dueling_DQN
from replayMemory import PrioritizedReplayBuffer, Transition

class DQNAgent(AgentWithConverter):
    def __init__(self, env, observation_space, action_space, args=None):
        super(DQNAgent, self).__init__(action_space, action_space_converter=IdToAct)
        self.env = env
        self.obs_space = observation_space
        # self.action_space = action_space
        # print('Filtering actions..')
        # self.action_space.filter_action(self._filter_action)
        # print('Done')
        self.obs_size = observation_space.size()
        self.action_size = action_space.size()
        print('obs space:{}; action space: {}'.format(self.obs_size, self.action_size))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = Dueling_DQN(self.obs_size, self.action_size).to(self.device)
        self.target_net = Dueling_DQN(self.obs_size, self.action_size).to(self.device)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 5000
        self.BATCH_SIZE = 128
        self.GAMMA = 0.98
        self.memory_size = 200000
        self.learning_rate = 1e-3
        self.n_epochs = 10000
        self.n_steps = 100

        self.memory = PrioritizedReplayBuffer(self.memory_size, './replay_buffer.pkl')
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

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
        """
        action = self.select_action(transformed_obs, steps_done)

        return action

    def select_action(self, obs, steps_done, test=False):
        """
        基于e-greedy策略选择动作
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * steps_done / self.EPS_DECAY)
        if sample > eps_threshold or test:
            with torch.no_grad():
                obs = torch.from_numpy(obs).to(self.device)
                # print(obs.shape)
                return self.policy_net(obs).max(1)[1].view(1, 1)

        return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        神经网络更新
        """
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        transitions, weights, batch_idxes = self.memory.sample(self.BATCH_SIZE)
        weights = torch.tensor(weights, device=self.device)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).type(FloatTensor)
        state_batch = torch.cat(batch.state).type(FloatTensor)
        action_batch = torch.cat(batch.action).type(LongTensor)
        reward_batch = torch.cat(batch.reward).type(FloatTensor)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        td_errors = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1),
                                     reduce=None) * weights
        new_priorities = np.abs(td_errors.cpu().detach().numpy()) + 1e-6
        self.memory.update_priorities(batch_idxes, new_priorities)
        td_errors = td_errors.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        td_errors.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return td_errors.item(), expected_state_action_values.mean()

    def train(self):
        """
        agent训练函数
        """
        total_steps = 0
        for epoch in range(self.n_epochs):
            total_reward = loss = train_loss = step = 0
            obs = self.convert_obs(self.env.reset())
            while True:
                step += 1
                action = self.my_act(obs, steps_done=total_steps)
                obs_next, reward, done, info = self.env.step(self.convert_act(action.item()))
                obs_next = self.convert_obs(obs_next)

                reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                done = torch.tensor([done], device=self.device, dtype=torch.float)
                obs_next_tensor = torch.from_numpy(obs_next).to(self.device)
                obs_tensor = torch.from_numpy(obs).to(self.device)

                if done:
                    obs_next = None
                    self.memory.push(obs_tensor.cpu(), action.cpu(), obs_next, reward.cpu(), done.cpu())
                else:
                    self.memory.push(obs_tensor.cpu(), action.cpu(), obs_next_tensor.cpu(), reward.cpu(), done.cpu())

                if len(self.memory) >= self.BATCH_SIZE:
                    loss, q_value = self.optimize_model()
                    total_steps += 1
                    train_loss += loss

                obs = obs_next
                total_reward += reward.item()
                # print('step: {} || reward: {:.3f} || train loss: {:.3f}'.format(step, reward.item(), loss))
                if done:
                    break
            print('--- epoch: [{}] || steps: [{}] || avg reward: [{:.3f}] || avg loss: [{:.3f}] ---'
                  .format(epoch+1, step, total_reward/step, train_loss/step))

    def save(self, path, epoch=0):
        """
        保存模型权重
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy_net.state_dict(),
        }, path)

    def load(self, path):
        """
        读取模型权重
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        epoch_end = checkpoint['epoch']

        return epoch_end
