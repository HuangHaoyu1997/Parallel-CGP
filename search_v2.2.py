'''
search oil mechs

'''
from postprocessing import *
import numpy as np
import time, os, random, shutil, pickle, gym, ray
from cmath import inf
from cgp import *
from copy import deepcopy
import matplotlib.pyplot as plt
from configuration import config
from paves.scenarios.oil_world.oil_world import Oil_World, s_g_example
from paves.scenarios.oil_world.config import Oil_Config
import warnings
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

@ray.remote
def rollout(env:Oil_World, policy):
    def func_wrapper(input):
        #####################################
        ## 调整这里
        #####################################
        # print(input)
        # close_clip的归一化方案
        input[0] = 1.0 if input[0] else 0.0
        input[3] = 1.0 if input[3] else 0.0
        input[5] /= 100
        input[7] /= 1000
        input[8] /= 1000
        
        # close的归一化方案
        # input[3] = 1.0 if input[3] else 0.0
        # input[5] /= 100
        # input[7] /= 1000
        # input[8] /= 1000
        # print(input,'\n')
        
        tmp = policy.eval(*input)
        # tmp = np.clip(tmp, a_min=0.0, a_max=1.0).tolist()
        return [tmp]
    
    env._add_mech(mech_name="new_mech",
                  func=func_wrapper,
                  ########################################
                  ## 调整这里
                  ########################################
                  source_nodes=[1,2,3,4,5,6,7,8,9,10,11],
                  target_nodes=[0])
    # ow._get_s_g_props_value()
    rewards = []
    for _ in range(config.rollout_episode):
        seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
        random.seed(seed)
        np.random.seed(seed)
        env.reset()
        # env.reset_2(0)
        for i in range(975):  # max 975
            obs1, obs2 = env._get_obs()
            a_l = []
            for j, enter in enumerate(env.enters):
                a = enter.rb_1_action(obs2[j][0], env.num_cycle, enter.type)
                a_l.append(a)
            env._set_action(a_l)
            # env._set_modulate_action(0.1)
            rew = env.step()
            obs1, obs2 = env._get_obs()
            obs1, obs2 = env._scale_obs(obs1, obs2)
        r1 = env._change_reward()
        rewards.append(r1)
    return np.mean(rewards)
    

pop = create_population(config.MU+config.LAMBDA, 
                        input_dim=11, 
                        out_dim=1,
                        fs=fs,
                        out_random_active=False)

# training
for g in range(config.N_GEN):
    tick = time.time()
    # rollout(env, pop[0])
    fit_list = [rollout.remote(env, p) for p in pop]
    fitness = ray.get(fit_list)
    for f,p in zip(fitness, pop):
        p.fitness = -f
    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time:', round(time.time()-tick, 2),'best fitness:', pop[0].fitness)
    with open(logdir,'a+') as f:
        f.write(str(g)+' time:'+str(round(time.time()-tick, 2))+',best fitness:'+str(pop[0].fitness)+'\n')

    if g % config.ckpt_freq == 0:
        with open(os.path.join(dir, run_time+'CGP-'+str(g)+'.pkl'), 'wb') as f:
            pickle.dump(pop, f)
            
    gg = extract_computational_subgraph(pop[0])
    visualize(g=gg, 
              to_file="./results/"+run_time+"_Oil_"+str(g)+".png", 
              operator_map=DEFAULT_SYMBOLIC_FUNCTION_MAP,
              input_names=["close_clip",  # v0
                           # "close",       # v1
                           "gdp",         # v2
                           "announce",    # v3
                           "need",        # v4
                           "weather",     # v5
                           "geo_risk",    # v6
                           "stock",       # v7
                           "t2t_price",   # v8
                           "open_price",  # v9
                           "pr_ratio",    # v10
                           "stage",       # v11
                           ],
                )
ray.shutdown()
