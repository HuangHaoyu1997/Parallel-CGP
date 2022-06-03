from email import policy
import pickle, gym, time
from cgp import *
import numpy as np

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

if __name__ == '__main__':
    with open('./results/CGP_LunarLanderContinuous-v2-79.pkl', 'rb') as f:
        pop = pickle.load(f)

    policy:Individual = pop[0]
    env = gym.make('LunarLanderContinuous-v2')
    
    test(env, policy)