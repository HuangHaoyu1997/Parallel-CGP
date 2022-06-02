import pickle, cma, copy, gym
from cgp import *

with open('./results/CGP_SP-29.pkl', 'rb') as f:
    pop:Individual = pickle.load(f)

print(pop[0].fitness)
pop[0]._determine_active_nodes()
weights, n_active = [], 0
for node in pop[0].nodes:
    if node.active:
        n_active += 1
        print(node.weights)
        weights.extend(node.weights)
print(weights, n_active)
es = cma.CMAEvolutionStrategy(
    x0=[1.0]*len(weights),
    sigma0=1.0,
    inopts={'popsize':7},
)
solution = es.ask()
print(solution[0])
individual:Individual = copy.deepcopy(pop[0])



# print([node.weights for node in individual.nodes])

def func(ind:Individual, solution):
    ind._determine_active_nodes()
    pointer = 0
    for node in ind.nodes:
        if node.active:
            node.weights = solution[pointer:pointer+node.arity]
            pointer += node.arity
    
    env = gym.make('Pendulum-v1')
    s = env.reset()
    done, rr = False, 0
    while not done:
        action = ind.eval(*s)
        s, r, done, _ = env.step(action)
        rr += r
    return rr
