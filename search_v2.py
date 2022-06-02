'''
多进程CGP算法训练Social Production policy
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
def func(env, ind:Individual):
    tick = time.time()
    reward = 0
    for _ in range(config.Epoch):
        s = env.reset()
        done = False
        while not done:
            action = ind.eval(*s)
            # action = 0 if action<0 else 1
            s, r, done, _ = env.step([action])
            reward += r
    reward /= config.Epoch
    ind.fitness = reward
    return reward

env = gym.make('Pendulum-v1')  #CartPole-v1
pop = create_population(config.MU+config.LAMBDA, input_dim=3, out_dim=1)
best_f = -inf
best_ff = -inf
best_ind = None

total_agent = config.MU + config.LAMBDA

# 开始搜索
for g in range(config.N_GEN):
    
    # 运行1代总时间
    tick = time.time()
    fit_list = [func.remote(env, p) for p in pop]
    fitness = ray.get(fit_list)
    for f,p in zip(fitness,pop):
        p.fitness = f

    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time for one generation:', time.time()-tick, pop[0].fitness)
    if g % 10 == 9:
        with open('./results/CGP_SP-'+str(g)+'.pkl','wb') as f:
            pickle.dump(pop,f)
ray.shutdown()

rr = 0
for i in range(100):
    r_e = 0
    done = False
    s = env.reset()
    while not done:
        action = pop[0].eval(*s)
        s, r, done, _ = env.step([action])
        r_e += r
    rr += r_e
    print(i, r_e)
print(rr/100)