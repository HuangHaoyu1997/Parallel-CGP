import pickle, cma, copy, gym, ray, time
from cgp import *
from configuration import config

# ray.init(num_cpus=config.n_process)

with open('./results/CGP_SP-79.pkl', 'rb') as f:
    pop = pickle.load(f)


pop[0]._determine_active_nodes()
weights, n_active = [], 0
for node in pop[0].nodes:
    if node.active:
        n_active += 1
        weights.extend(node.weights)

individual:Individual = copy.deepcopy(pop[0])
es = cma.CMAEvolutionStrategy(
    x0=[-1.]*len(weights),
    sigma0=1.0,
    inopts={'popsize':20},
)

@ray.remote
def func(ind:Individual, solution):
    ind._determine_active_nodes()
    pointer = 0
    for node in ind.nodes:
        if node.active:
            node.weights = solution[pointer:pointer+node.arity]
            pointer += node.arity
    reward = 0
    for i in range(config.Epoch):
        env = gym.make('Pendulum-v1')
        env.seed(int(time.time()*1000))
        s = env.reset()
        done, rr = False, 0
        while not done:
            action = ind.eval(*s)
            s, r, done, _ = env.step([action])
            rr += r
            reward += r
    return -reward/config.Epoch
print('best fitness before weight search:', pop[0].fitness)
print('start training')
for i in range(config.N_GEN):
    solutions = es.ask()
    fit = [func.remote(individual, solution) for solution in solutions]
    fitness = ray.get(fit)
    es.tell(solutions, fitness)
    best_f, best_x = -es.best.f, es.best.x
    print(i, best_f)

