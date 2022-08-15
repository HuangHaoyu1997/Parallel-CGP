from email import policy
import pickle, gym, time
from cgp import *
import numpy as np
from paves.scenarios.oil_world.oil_world import Oil_World, s_g_example
from paves.scenarios.oil_world.config import Oil_Config
from utils import *
from function import protected_div
import operator as op

def test(env, policy):
    rr = []
    for i in range(100):
        done, reward = False, 0
        env.seed(int(time.time())*1000)
        s = env.reset()
        while not done:
            action = policy.eval(*s)
            s, r, done, _ = env.step(action)
            reward += r
        # print(i, reward)
        rr.append(reward)
    print(np.mean(rr), np.std(rr))

def rollout(env:Oil_World):
    def func_wrapper(input):
        #####################################
        ## 调整这里
        #####################################
        # print('aa',input)
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
        
        
        t2t = float(input[0])
        open = float(input[1])
        sub = -(t2t - open)
        div = protected_div(t2t, sub)
        result = min2(op.abs(div), op.abs(sub)) + sub
        # tmp = np.clip(tmp, a_min=0.0, a_max=1.0).tolist()
        return [result]
    env.reset()
    env._add_mech(mech_name="new_mech",
                  func=func_wrapper,
                  ########################################
                  ## 调整这里
                  ########################################
                  source_nodes=[8,9],
                  target_nodes=[1])
    rewards, ss = [], []
    for _ in range(5): # config.rollout_episode
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
        
        if zeros >= 800:
            r1 = 10000
            rewards.append(r1)
            continue
        # 第一种损失函数-标准差
        r1 = std(env.simu_price)
        # 第二种损失函数-布林带平均宽度
        Up, Low = bollinger_bands(env.simu_price, 10, 2)
        r2 = bollinger_reward(Up, Low)
        rewards.append(r1)
        ss.append(env.simu_price)
        
    return np.mean(rewards), np.array(ss).mean(0), env.real_price

if __name__ == '__main__':
    # with open('./results/CGP_BipedalWalker-v3-60.pkl', 'rb') as f:
    #     pop = pickle.load(f)

    # policy:Individual = pop[0]
    # env = gym.make('BipedalWalker-v3') # 'LunarLanderContinuous-v2'
    # test(env, policy)
    
    # creating env
    s_g = s_g_example()
    OW_config = Oil_Config()
    env = Oil_World(OW_config, 1000, 5, 3, 1, 20, 20, 10, smeg_graph=s_g)
    env._change_price_para([0.019,0.038,0.124,0.09,0.0044,0.0168,0.0125])
    re, s, r = rollout(env)
    ss_before = plot_fit_visual()
    print(re, np.std(s), np.std(r))
    plt.plot(s, marker='s', markersize=4,)
    plt.plot(r, marker='s', markersize=4,)
    plt.plot(ss_before, marker='s', markersize=4,)
    plt.legend(['simu', 'real', 'simu_before'])
    
    plt.rcParams["font.size"] = 12
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel('Simulation Period / Year')
    plt.ylabel('Oil Price')
    # plt.ylim(bottom=0)
    ticks = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60]
    labels = ['2006/1', '2007/1', '2008/1', '2009/1', 
              '2010/1', '2011/1', '2012/1', '2013/1', 
              '2014/1', '2015/1', '2016/1', '2017/1',
              '2018/1', '2019/1', '2020/1', '2021/1',]
    plt.xticks(
        ticks,
        labels,
        rotation=45,
    )
    
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    
    plt.savefig('fluctuations.png')
