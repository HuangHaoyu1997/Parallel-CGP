"""
利用测试数据集，得到理想的符号表达式
"""
from itertools import count
from postprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import time, os, random, shutil, pickle, gym, ray
from cgp import *
from configuration import config
import warnings
warnings.filterwarnings('ignore')

def load_testset():
    with open('label_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def get_testset():
    open_price = np.arange(200, 1500, 10)
    t2t_price_up = np.zeros_like(open_price)
    t2t_price_low = np.zeros_like(open_price)
    
    dataset = []
    count= 0
    for i in range(len(open_price)):
        if np.random.rand() < 0.5:
            factor = np.random.uniform(1, 1.02)
            t2t_price_up[i] = open_price[i] * factor
            dataset.append([open_price[i], t2t_price_up[i], 1])
            count += 1
        else:
            factor = np.random.uniform(1.02, 1.5)
            t2t_price_up[i] = open_price[i] * factor
            dataset.append([open_price[i], t2t_price_up[i], 0])
            
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.98, 1)
            t2t_price_low[i] = open_price[i] * factor
            dataset.append([open_price[i], t2t_price_low[i], 1])
            count += 1
        else:
            factor = np.random.uniform(0.5, 0.98)
            t2t_price_low[i] = open_price[i] * factor
            dataset.append([open_price[i], t2t_price_low[i], 0])
        
    return dataset, count

@ray.remote
def rollout(dataset, policy):
    count = 0
    for d in dataset:
        # input = [d[0], d[1]]
        # label = d[2]
        input = [d[0], d[1], d[2], d[3]]
        label = d[4]
        output = policy.eval(*input)
        if label == output:
            count += 1
    return count / len(dataset)


if __name__ == '__main__':
    
    ray.init(num_cpus=config.n_process)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    run_time = (time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))[:19]
    env_name = 'Oil_dataset/' # 'CartPole-v1' # 'CartPoleContinuous'
    dir = './results/CGP_' + env_name
    logdir = dir + 'log-'+run_time+'.txt'
    
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir+'imgs/'+run_time+'/'):
        os.mkdir(dir+'imgs/'+run_time+'/')
    if not os.path.exists(dir+'pkls/'+run_time+'/'):
        os.mkdir(dir+'pkls/'+run_time+'/')
    pop = create_population(config.MU+config.LAMBDA, 
                            input_dim=4, 
                            out_dim=1,
                            fs=fs,
                            out_random_active=False)
    
    # data, count = get_testset()
    data = load_testset()
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
        with open(logdir,'a+') as f:
            f.write(str(g)+' time:'+str(round(time.time()-tick, 2))+',best fitness:'+str(pop[0].fitness)+'\n')
        if g % config.ckpt_freq == 0:
            with open(dir+ 'pkls/' + run_time + '/' + str(g) + '.pkl', 'wb') as f:
                pickle.dump(pop, f)
        
        gg = extract_computational_subgraph(pop[0])
        visualize(g=gg, 
                to_file=dir+'imgs/'+run_time+'/'+str(g)+'_'+str(pop[0].fitness)+".png", 
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
                                "t2t_deal_amount", # v13
                                "t2t_deal_money", #v14
                                ],
                    )
        if pop[0].fitness == 1.0:
            print('done!')
            break
    ray.shutdown()