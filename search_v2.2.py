'''
search oil mechs

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
# env._change_price_para([0.019,0.038,0.124,0.09,0.0044,0.0168,0.0125])
# prop_flags = [1]*77
# env._change_smeg(prop_flags, [0, 0, 1, 1, 1, 1, 1, 1, 0])

@ray.remote
def rollout(env:Oil_World, policy):
    def func_wrapper(input):
        #####################################
        ## 调整这里
        #####################################
        print('aa',input)
        # close_clip的归一化方案
        # input[0] = 1.0 if input[0] else 0.0
        # input[3] = 1.0 if input[3] else 0.0
        # input[6] /= 1000
        # input[7] /= 1000
        
        # close的归一化方案
        # input[3] = 1.0 if input[3] else 0.0
        # input[5] /= 100
        # input[7] /= 1000
        # input[8] /= 1000
        # print(input, '\n')
        
        tmp = policy.eval(*input)
        # print(tmp)
        # tmp = np.clip(tmp, a_min=0.0, a_max=1.0).tolist()
        return [tmp]
    
    env._add_mech(mech_name="new_mech",
                  func=func_wrapper,
                  ########################################
                  ## 调整这里
                  ########################################
                  source_nodes=[8,9],
                #   source_nodes=[2,3,9,10,12,13,14,15,18],
                  target_nodes=[11])
    rewards = []
    for _ in range(config.rollout_episode):
        seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
        random.seed(seed)
        np.random.seed(seed)
        
        
        env.reset()
        # env.reset_2(0)
        zeros = 0
        for i in range(960):  # max 975
            obs1, obs2 = env._get_obs()
            a_l = []
            for j, enter in enumerate(env.enters):
                a = enter.rb_1_action(obs2[j][0], env.num_cycle, enter.type, env.price_para_list)
                a_l.append(a)
            env._set_action(a_l)
            # print(env._get_s_g_props_value())
            rew = env.step()
            obs1, obs2 = env._get_obs()
            obs1, obs2 = env._scale_obs(obs1, obs2)
            if env.market.order_books[0].t2t_deal_amount == 0:
                zeros += 1
        
        if zeros > 192:
            r1 = 10000
            rewards.append(r1)
            continue
        # 第一种损失函数-标准差
        r1 = std(env.simu_price)
        # 第二种损失函数-布林带平均宽度
        Up, Low = bollinger_bands(env.simu_price, 10, 2)
        r2 = bollinger_reward(Up, Low)
        rewards.append(r2)
        
        
    return np.mean(rewards)
    

pop = create_population(config.MU+config.LAMBDA, 
                        input_dim=9, 
                        out_dim=1,
                        fs=fs,
                        out_random_active=False)

# training
for g in range(config.N_GEN):
    tick = time.time()
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
              to_file="./results/"+run_time+"_Oil_"+str(g)+'_'+str(pop[0].fitness)+".png", 
              operator_map=DEFAULT_SYMBOLIC_FUNCTION_MAP,
              input_names=[
                            # "close_clip",    # v0
                            # "close",        # v1
                            # "gdp",          # v2
                            # "announce",     # v3
                            # "need",         # v4
                            # "weather",      # v5
                            # "geo_risk",     # v6
                            # "stock",        # v7
                            "t2t_price",    # v8
                            "open_price",   # v9
                            # "pr_ratio",     # v10
                            # "stage",        # v11
                            # "order_type",   # v12
                            ],
                )
ray.shutdown()
