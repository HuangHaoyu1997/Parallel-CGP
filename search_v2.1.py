'''
using evolution strategy and CGP to optimize symbol policy
for Oil World Tasks

'''
from multiprocessing import Process
import numpy as np
import time, os, random, shutil, pickle, gym, ray
from cmath import inf
from cgp import *
from copy import deepcopy
import matplotlib.pyplot as plt
from configuration import config
from paves.scenarios.oil_world.oil_world import Oil_World
from paves.scenarios.oil_world.config import Oil_Config
import warnings
warnings.filterwarnings('ignore')

ray.init(num_cpus=config.n_process)
np.random.seed(config.seed)
random.seed(config.seed)

run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
env_name = 'Oil' # 'CartPole-v1' # 'CartPoleContinuous'
logdir = './results/log-'+env_name+'-'+run_time+'.txt'

OW_config = Oil_Config()
env = Oil_World(OW_config, 1000, 5, 3, 3, history_len=5)
env.reset()
inpt_dim = len(env._get_modulate_obs())
out_dim = 1

dir = './results/CGP_' + env_name
if not os.path.exists(dir):
    os.mkdir(dir)

@ray.remote
def rollout(env, policy):
    def reward_func(env):
        if len(env.market.oil_price_his) <= 1:
            return 0
        if env.time % env.day_rounds != 0:
            return 0
        std = np.std(env.market.oil_price_his)
        if std < 40 and std > 60:
            return -1
        else:
            return 1
    
    rewards = []
    for _ in range(config.rollout_episode):
        seed = int(str(time.time()).split('.')[1]) # if not test else config.seed
        random.seed(seed)
        np.random.seed(seed)
        env.reset()
        reward = 0
        state = env._get_modulate_obs()
        for _ in range(config.n_steps):
            action = policy.eval(*state)
            env._set_modulate_action(action)
            reward += reward_func(env)
            state = env._get_modulate_obs()
        rewards.append(reward)
    return np.mean(rewards)
    

pop = create_population(config.MU+config.LAMBDA, input_dim=8, out_dim=2)
best_f = -inf
best_ff = -inf
best_ind = None

# training
for g in range(config.N_GEN):
    tick = time.time()
    fit_list = [rollout.remote(env, p) for p in pop]
    fitness = ray.get(fit_list)
    for f,p in zip(fitness, pop):
        p.fitness = f
    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time:', round(time.time()-tick, 2),'best fitness:', pop[0].fitness)
    with open(logdir,'a+') as f:
        f.write(str(g)+' time:'+str(round(time.time()-tick, 2))+',best fitness:'+str(pop[0].fitness)+'\n')

    if g % config.ckpt_freq == 0:
        with open(os.path.join(dir, 'CGP-'+str(g)+'.pkl'), 'wb') as f:
            pickle.dump(pop, f)
ray.shutdown()
