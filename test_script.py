from cgp import *
from configuration import config
from function import fs
pop = create_population(config.MU+config.LAMBDA, 
                        input_dim=12, 
                        out_dim=3,
                        fs=fs,
                        out_random_active=True
                        )

node:Node = pop[0].nodes[-2]
print(node.i_inputs)