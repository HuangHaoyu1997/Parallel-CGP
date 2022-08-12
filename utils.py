import numpy as np
import matplotlib.pyplot as plt
from paves.scenarios.oil_world.oil_world import Oil_World, s_g_example
from paves.scenarios.oil_world.config import Oil_Config

def simple_moving_avg(seq, N=5):
    l_seq = len(seq)
    avg = []
    for i in range(l_seq - N):
        mean = np.mean(seq[i:i+N])
        avg.append(mean)
    return np.array(avg)

def exponential_moving_avg(seq, weight=0.5):
    l_seq = len(seq)
    avg = []
    avg.append(seq[0])
    for i in range(l_seq-1):
        avg.append(avg[-1]*weight + (1-weight)*seq[i+1])
    return avg

def std(seq):
    tmp = np.std(seq)
    tmp = tmp if tmp>0 else 10000
    return tmp

def bollinger_bands(seq, N, K):
    def N_period_std(seq, N):
        return np.array([np.std(seq[i:i+N]) for i in range(len(seq)-N)])
    N_avg = simple_moving_avg(seq, N)
    N_std = N_period_std(seq, N)
    return N_avg + K * N_std, N_avg - K * N_std
    
def bollinger_reward(Up, Low):
    return np.mean(Up - Low)

def plot_fit_visual():
    s_g = s_g_example()
    con = Oil_Config()
    ow = Oil_World(con, 1000, 5, 3, 1, 20, 20, 10, smeg_graph=s_g)
    ow._change_price_para([0.019, 0.038, 0.124, 0.09, 0.0044, 0.0168, 0.0125])
    ow._change_smeg([1]*12, [0,0,1,1,1,1,1,1,0])
    ss, rr = [], []
    for j in range(1):
        np.random.seed(j+1234)
        ow.reset()
        for i in range(960):  # max 975
            a_l = []
            obs1, obs2 = ow._get_obs()
            for j, enter in enumerate(ow.enters):
                a = enter.rb_1_action(obs2[j][0], ow.num_cycle, enter.type,ow.price_para_list)
                a_l.append(a)
            ow._set_action(a_l)
            ow._set_modulate_action(0.1)
            rew = ow.step()

        s = np.array(ow.simu_price)
        r = np.array(ow.real_price)
        ss.append(s)
        rr.append(r)
    ss = np.array(ss).mean(0)
    rr = np.array(rr).mean(0)
    BBup, BBlow = bollinger_bands(ss, 10, 2)
    plt.plot(ss, marker='s', markersize=4,)
    plt.plot(rr, marker='s', markersize=4,)
    plt.plot(BBup, )
    plt.plot(BBlow, )
    plt.rcParams["font.size"] = 12
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.legend(['simu','real', 'bollinger upper band', 'bollinger lower band'])
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
    # plt.xticks([i for i in range(64)], 
    #            ['2006/1', '2006/2', '2006/3', '2006/4', '2007/1', '2007/2', '2007/3', '2007/4',
    #             '2008/1', '2008/2', '2008/3', '2008/4', '2009/1', '2009/2', '2009/3', '2009/4',
    #             '2010/1', '2010/2', '2010/3', '2010/4', '2011/1', '2011/2', '2011/3', '2011/4',
    #             '2012/1', '2012/2', '2012/3', '2012/4', '2013/1', '2013/2', '2013/3', '2013/4',
    #             '2014/1', '2014/2', '2014/3', '2014/4', '2015/1', '2015/2', '2015/3', '2015/4',
    #             '2016/1', '2016/2', '2016/3', '2016/4', '2017/1', '2017/2', '2017/3', '2017/4',
    #             '2018/1', '2018/2', '2018/3', '2018/4', '2019/1', '2019/2', '2019/3', '2019/4',
    #             '2020/1', '2020/2', '2020/3', '2020/4', '2021/1', '2021/2', '2021/3', '2021/4',],

    #            ['2006/1', '', '', '', '2007/1', '', '', '',
    #             '2008/1', '', '', '', '2009/1', '', '', '',
    #             '2010/1', '', '', '', '2011/1', '', '', '',
    #             '2012/1', '', '', '', '2013/1', '', '', '',
    #             '2014/1', '', '', '', '2015/1', '', '', '',
    #             '2016/1', '', '', '', '2017/1', '', '', '',
    #             '2018/1', '', '', '', '2019/1', '', '', '',
    #             '2020/1', '', '', '', '2021/1', '', '', '',],
    #            # [str(i) for i in range(2006,2022)],
    #            rotation=45,
    #            )
    plt.savefig('best_fit_6mech.svg', dpi=300, format='svg', bbox_inches='tight')


if __name__ == '__main__':
    # seq = np.random.rand(100) + np.sin(np.arange(100))*2
    # result = simple_moving_avg(seq)
    # plt.plot(result)
    # plt.plot(seq)
    # plt.savefig('test.png')
    # plt.show()
    plot_fit_visual()