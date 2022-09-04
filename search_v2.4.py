'''
收集数据，通过多次仿真
然后对数据打标签，得到训练CGP的数据集

'''
from postprocessing import *
import numpy as np
import time, os, random, shutil, pickle, gym, ray
from cmath import inf
from cgp import *
import matplotlib.pyplot as plt
from configuration import config
from paves.scenarios.oil_world.oil_world import Oil_World, s_g_example
from paves.scenarios.oil_world.config import Oil_Config
import warnings

from utils import bollinger_bands, bollinger_reward, std
warnings.filterwarnings('ignore')

ray.init(num_cpus=config.n_process)
np.random.seed(config.seed)
random.seed(config.seed)

run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
env_name = 'Oil' # 'CartPole-v1' # 'CartPoleContinuous'
logdir = './results/log-'+env_name+'-'+run_time+'.txt'

dir = './results/CGP_' + env_name
if not os.path.exists(dir):
    os.mkdir(dir)

# creating env
s_g = s_g_example()
OW_config = Oil_Config()
env = Oil_World(OW_config, 1000, 5, 3, 1, 20, 20, 10, smeg_graph=s_g)
env._change_price_para([0.019,0.038,0.124,0.09,0.0044,0.0168,0.0125])


def rollout(env:Oil_World, ):
    
    seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
    random.seed(seed)
    np.random.seed(seed)
    
    env.reset()
    data = []
    for i in range(960):  # max 975
        obs1, obs2 = env._get_obs()
        a_l = []
        for j, enter in enumerate(env.enters):
            a = enter.rb_1_action(obs2[j][0], env.num_cycle, enter.type, env.price_para_list)
            a_l.append(a)
        env._set_action(a_l)
        # print(env._get_s_g_props_value())
        rew = env.step()
        value = env._get_s_g_props_value()
        data.append([value[8], value[9], value[13], value[14]])
        obs1, obs2 = env._get_obs()
        obs1, obs2 = env._scale_obs(obs1, obs2)

    return env.real_price, env.simu_price, data
    

# real1, simu1 = rollout(env)
# real2, simu2 = rollout(env)
# real3, simu3 = rollout(env)
# real4, simu4 = rollout(env)
# real5, simu5 = rollout(env)
# plt.plot(real1); plt.plot(real2); plt.plot(real3); plt.plot(real4); plt.plot(real5)
# plt.plot(simu1); plt.plot(simu2); plt.plot(simu3); plt.plot(simu4); plt.plot(simu5)
# plt.legend(['real','real','real','real','real', 'simu','simu','simu','simu','simu'])
# plt.show()
data = []
for i in range(2):
    _, _, d = rollout(env)
    data.extend(d)
print(len(data))
labeled_data = []
for d in data:
    # d[0] t2t_price
    # d[1] open_price
    if d[0] / d[1] - 1 >= -0.02 and d[0] / d[1] - 1 <= 0.02:
        d.append(1)
    else:
        d.append(0)
    labeled_data.append(d)
with open('label_data.pkl', 'wb') as f:
    pickle.dump(labeled_data, f)

