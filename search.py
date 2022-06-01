'''
多进程CGP算法训练Social Production policy
'''
from multiprocessing import Process
import numpy as np
import time, os, random, shutil, pickle, gym
from cmath import inf
from cgp import *
import matplotlib.pyplot as plt
from configuration import config
import warnings

warnings.filterwarnings('ignore')

np.random.seed(config.seed)
random.seed(config.seed)

def func(idx, env, pop):
    '''
    子进程所执行的函数
    idx: 进程号
    pop: 种群
    '''
    reward_pop = []
    tick = time.time()
    for p in pop: # 遍历每个individual
        reward = 0
        for _ in range(config.Epoch):
            s = env.reset()
            done = False
            while not done:
                action = p.eval(*s)
                # action = 0 if action<0 else 1
                s, r, done, _ = env.step([action])
                reward += r
        reward /= config.Epoch
        p.fitness = reward
        reward_pop.append(reward)
    with open('./tmp/'+str(idx)+'.pkl','wb') as f:
        pickle.dump(reward_pop,f)
    # print(idx, (time.time()-tick) / (len(pop)*config.Epoch), ' finished!')

env = gym.make('Pendulum-v1')  #CartPole-v1
pop = create_population(config.MU+config.LAMBDA, input_dim=3, out_dim=1)
best_f = -inf
best_ff = -inf
best_ind = None

total_agent = config.MU + config.LAMBDA
agent_p = int(total_agent/config.n_process) # 平均每个process分到的agent数量

# 开始搜索
for g in range(config.N_GEN):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp/')
    
    # 运行1代总时间
    tick = time.time()
    process = []
    
    for i in range(config.n_process):
        process.append(Process(target=func, args=(i, env, pop[i*agent_p:(i+1)*agent_p])))

    [p.start() for p in process]
    [p.join() for p in process]
    
    fitness = []
    for i in range(config.n_process):
        with open('./tmp/'+str(i)+'.pkl','rb') as f:
            data = pickle.load(f)
            fitness.extend(data)
    for f,p in zip(fitness,pop):
        p.fitness = f

    fitness = np.array(fitness)
    idx = fitness.argsort()[::-1][0:config.MU]
    shutil.rmtree('./tmp/',True)
    
    
    pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
    print(g,'time for one generation:', time.time()-tick, pop[0].fitness)
    #if pop[0].fitness > config.solved:
    if g % 10 == 9:
        with open('./results/CGP_SP-'+str(g)+'.pkl','wb') as f:
            pickle.dump(pop,f)


rr = 0
for i in range(100):
    r_e = 0
    done = False
    s = env.reset()
    while not done:
        action = pop[0].eval(*s)
        action = np.min(np.max(action, 0), 200)
        # action = np.random.choice(4,p=action)
        s, r, done, _ = env.step(action)
        r_e += r
    rr += r_e
    print(i, r_e)
print(rr/100)