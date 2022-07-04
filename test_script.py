'''
草稿纸
草稿纸
草稿纸
草稿纸
'''
from cgp import *
from configuration import config
from function import fs
from multiprocessing import Process
import numpy as np
import time, os, random, shutil, pickle, gym, ray
from cmath import inf
from cgp import *
from copy import deepcopy
import matplotlib.pyplot as plt
from configuration import config
from paves.scenarios.oil_world.oil_world import Oil_World,s_g_example
from paves.scenarios.oil_world.config import Oil_Config
import warnings
warnings.filterwarnings('ignore')

pop = create_population(config.MU+config.LAMBDA, 
                        input_dim=12, 
                        out_dim=3,
                        fs=fs,
                        out_random_active=False,
                        )
pop[0]._determine_active_nodes()
node1:Node = pop[0].nodes[-1]
node2:Node = pop[0].nodes[-2]
node3:Node = pop[0].nodes[-3]
print(node1.active, node2.active, node3.active, pop[0].n_active)


def test_func(source_nodes_value_list):
    output_list=[0.001]
    return output_list

s_g = s_g_example()
np.random.seed(900)
con = Oil_Config()
ow = Oil_World(con, 1000, 5, 3, 1, 20, 20, 10, smeg_graph=s_g)
# new_mech_example
ow._add_mech(mech_name="new_mech",func=test_func,source_nodes=[1,2,3],target_nodes=[0])
ow.reset()
input = ow._get_s_g_props_value()
print(pop[0].eval(*input))