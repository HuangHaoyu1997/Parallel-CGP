import numpy as np
import matplotlib.pyplot as plt


def simple_moving_avg(seq, N=5):
    l_seq = len(seq)
    avg = []
    for i in range(l_seq - N):
        mean = np.mean(seq[i:i+N])
        avg.append(mean)
    return avg

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

if __name__ == '__main__':
    seq = np.random.rand(100) + np.sin(np.arange(100))*2
    result = simple_moving_avg(seq)
    plt.plot(result)
    plt.plot(seq)
    plt.savefig('test.png')
    plt.show()