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


def select_obs(obs_env):
    """
    :param obs_env: obs in grid2op object form
    :return: selected obs (numpy array)
    """

    selected_obs = []
    return selected_obs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    env_name = "l2rpn_neurips_2020_track1_small"
    backend = LightSimBackend()
    env = grid2op.make(env_name, reward_class=L2RPNReward, backend=backend)
    print('env creating:', env_name)
    env.seed(123)

    action_space = env.action_space
    observation_space = env.observation_space

    # observation space
    do_nothing_action = env.action_space({})

    features = []
    steps_to_done = []
    # THRESHOLD = 20

    for t in range(1, 20000):
        env_id = t * 2 + 1
        env.set_id(env_id)
        env.reset()
        skip = random.randint(1, 200)
        env.fast_forward_chronics(skip)

        queue_obs = []
        for step in range(0, 2100):
            obs_next_env, reward, done, info = env.step(do_nothing_action)
            queue_obs.append(obs_next_env)
            if done:
                if step > 20:
                    # 选择两条安全的
                    A_step_to_done = randint(0, step-20)
                    B_step_to_done = randint(0, step-20)
                    # 选择两条可能危险的
                    C_step_to_done = randint(step-20+1, step-1)
                    D_step_to_done = randint(step-20+1, step-1)
                    # 保存step和obs
                    steps_to_done.append(step - A_step_to_done)
                    steps_to_done.append(step - B_step_to_done)
                    steps_to_done.append(step - C_step_to_done)
                    steps_to_done.append(step - D_step_to_done)

                    features.append(queue_obs[A_step_to_done].to_vect())
                    features.append(queue_obs[B_step_to_done].to_vect())
                    features.append(queue_obs[C_step_to_done].to_vect())
                    features.append(queue_obs[D_step_to_done].to_vect())

                else:
                    C_step_to_done = randint(0, step - 1) - 1
                    D_step_to_done = randint(0, step - 1) - 1

                    steps_to_done.append(step - C_step_to_done)
                    steps_to_done.append(step - D_step_to_done)

                    features.append(queue_obs[C_step_to_done].to_vect())
                    features.append(queue_obs[D_step_to_done].to_vect())
                break

        if t % 10 == 0:
            print("已完成 {} 份data, 记录样本 {} 条".format(t, len(steps_to_done)))

    features = np.array(features)
    steps_to_done = np.array(steps_to_done)
    np.save('features.npy', features)
    np.save("steps.npy", steps_to_done)
    print("保存成功")


