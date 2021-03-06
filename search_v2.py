'''
Multiprocess CGP for gym classical control
'''
from multiprocessing import Process
import numpy as np
import time, os, random, shutil, pickle, gym, ray
from cmath import inf
from cgp import *
import matplotlib.pyplot as plt
from configuration import config
import warnings
warnings.filterwarnings('ignore')

ray.init(num_cpus=config.n_process)
np.random.seed(config.seed)
random.seed(config.seed)

@ray.remote
def rollout(env, ind:Individual):
    tick = time.time()
    reward = 0
    for _ in range(config.rollout_episode):
        env.seed(int(time.time()*1000))
        s = env.reset()
        done = False
        while not done:
            action = ind.eval(*s)
            # action = 0 if action<0 else 1
            s, r, done, _ = env.step(action)
            reward += r
    reward /= config.rollout_episode
    return reward

env_name= 'BipedalWalker-v3' # 'CartPole-v1'  'Pendulum-v1'  'LunarLanderContinuous-v2'
env = gym.make(env_name)
pop = create_population(config.MU+config.LAMBDA, input_dim=24, out_dim=4)
with open('./results/CGP_BipedalWalker-v3-120.pkl', 'rb') as f:
    pop = pickle.load(f)

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
    if g % config.ckpt_freq == 0:
        with open('./results/CGP_'+env_name+'-'+str(g)+'.pkl','wb') as f:
            pickle.dump(pop, f)
ray.shutdown()

rr = 0
for i in range(100):
    r_e = 0
    done = False
    s = env.reset()
    while not done:
        action = pop[0].eval(*s)
        s, r, done, _ = env.step(action)
        r_e += r
    rr += r_e
    print(i, r_e)
print(rr/100)