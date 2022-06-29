class config:
    seed = 123
    N_COLS = 20
    LEVEL_BACK = 20
    Verbose = False
    MU = 5
    LAMBDA = 200 - MU
    n_process = 40
    N_GEN = 1000
    solved = 300 # for BipedalWalker-v3
    rollout_episode = 20
    MUT_PB = 0.1
    n_steps = 200
    ckpt_freq = 10
    sigma0 = 0.2 # for cma initial sigma
    popsize = 30 # for cma population
class BipedalWalkerconfig:
    seed = 123
    N_COLS = 100
    LEVEL_BACK = 100
    Verbose = False
    MU = 5
    LAMBDA = 200 - MU
    n_process = 40
    N_GEN = 1000
    solved = 300 # for BipedalWalker-v3
    rollout_episode = 20
    MUT_PB = 0.1
    n_steps = 200
    ckpt_freq = 10
    sigma0 = 0.2 # for cma initial sigma
    popsize = 30 # for cma population
LunarLanderContinuousConfig = config()
LunarLanderContinuousConfig.N_COLS = 30
LunarLanderContinuousConfig.LEVEL_BACK = 30
LunarLanderContinuousConfig.MUT_PB = 0.15


