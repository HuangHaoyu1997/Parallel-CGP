'''
2023年3月11日04:35:33
测试向量化符号集的可用性
'''

from postprocessing import *
import numpy as np
import time, os, random, shutil, pickle, ray
from cgp import *
import matplotlib.pyplot as plt
from configuration import config
import warnings
warnings.filterwarnings('ignore')


@ray.remote
def rollout(dataset, policy):
    x, y, z = dataset
    count = 0
    for xi,yi,zi in zip(x,y,z):
        # input = [d[0], d[1]]
        # label = d[2]
        input = [xi, yi]
        output = policy.eval(*input)
        if type(output)!=float:
            continue
        print(zi, output, zi == output)
        if zi == output:
            count += 1
    return count / len(x)

def get_data():
    num = 100
    dim = 10
    x, y, z = [], [], []
    for i in range(num):
        xi = np.random.rand(dim,)
        x.append(xi)
        yi = xi * (np.random.uniform([-0.15]*dim, [0.15]*dim) + 1)
        y.append(yi)
        
        zi = np.sum(np.sign(np.maximum((yi-xi)/xi-0.1, 0))*xi)
        z.append(zi)
        
        
    return [x, y, z]

if __name__ == '__main__':
    ray.init(num_cpus=config.n_process)
    np.random.seed(config.seed)
    random.seed(config.seed)

    run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
    env_name = 'Test' # 'CartPole-v1' # 'CartPoleContinuous'
    logdir = './results/log-'+env_name+'-'+run_time+'.txt'

    dir = './results/CGP_' + env_name
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    pop = create_population(config.MU+config.LAMBDA, 
                            input_dim=2, 
                            out_dim=1,
                            fs=new_functions,
                            out_random_active=False)
    data = get_data()
    # training
    for g in range(config.N_GEN):
        tick = time.time()
        fit_list = [rollout.remote(data, p) for p in pop]
        fitness = ray.get(fit_list)
        # 优化方向是最大化
        for f, p in zip(fitness, pop):
            p.fitness = f
        pop = evolve(pop, config.MUT_PB, config.MU, config.LAMBDA)
        print(g, 'time:', round(time.time()-tick, 2),'best fitness:', pop[0].fitness)
        # with open(logdir,'a+') as f:
        #     f.write(str(g)+' time:'+str(round(time.time()-tick, 2))+',best fitness:'+str(pop[0].fitness)+'\n')
