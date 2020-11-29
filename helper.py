import grid2op

track = 1
if track == 1:
    env_name = "l2rpn_neurips_2020_track1_small"
else:
    env_name = "l2rpn_neurips_2020_track2_small"
print('env name:', env_name)
env = grid2op.make(env_name)
obs = env.reset()

print("{}年{}月{}日{}时{}分|周{}".format(obs.year, obs.month, obs.day, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week))
print("变电站数量: {}".format(obs.n_sub))
print("发电机数量: {}".format(obs.n_gen))
print("负荷数量: {}".format(obs.n_load))
print("电线数量: {}".format(obs.n_line))
print("连接到各个变电站的元件数量: {}".format(obs.sub_info))
print("所有元件数量: {} (负荷+发电机+电线始端+电线终端)".format(obs.dim_topo))

action_space = env.action_space
obs_space = env.observation_space
print('observation 总数：', env.observation_space.n)
print('action 总数：', env.action_space.n)
print('--- observation space ---')
obs_obj = obs_space.subtype()
obs_obj_shape = obs_obj.shape()
obs_obj_dtype = obs_obj.dtype()
obs_obj_attr = obs_obj.attr_list_vect
obs_obj_dict = {}
for i in range(len(obs_obj_attr)):
    name = obs_obj_attr[i]
    shape = obs_obj_shape[i]
    dtype = obs_obj_dtype[i]
    obs_obj_dict[name] = [shape, dtype]
    # print(name, shape, dtype)
    print('{:25} {:3}\t'.format(name, shape), dtype)
# 每个observation属性的可在官方技术文档查询 https://grid2op.readthedocs.io/en/latest/observation.html')

# print(obs.topo_vect)  # 每个元件所连接的bus编号
# print(obs.timestep_overflow)  # 每条线路（如果）过载后已经持续的步长
# print(obs.time_before_cooldown_line)  # 每条线路（如果）过载后剩余的冷却步长（还差几步允许重连）
# print(action_space())  # action_space同名函数可以打印出具体改变了哪些电网参数

print('\n--- action space ---')
action_obj = action_space.subtype()
action_obj_shape = action_obj.shape()
action_obj_dtype = action_obj.dtype()
action_obj_attr = action_obj.attr_list_vect
action_obj_dict = {}
for i in range(len(action_obj_attr)):
    name = action_obj_attr[i]
    shape = action_obj_shape[i]
    dtype = action_obj_dtype[i]
    action_obj_dict[name] = [shape, dtype]
    # print(name, shape, dtype)
    print('{:25} {:3}\t'.format(name, shape), dtype)