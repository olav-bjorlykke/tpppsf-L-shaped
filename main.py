import copy

from model import Model
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import numpy as np
import initialization.parameters
from initialization.input_data import InputData
from initialization.site_class import Site
import matplotlib.pyplot as plt
import initialization.configs as configs
import time
from cg_master_problem import CGMasterProblem
import initialization.sites as sites
import initialization.parameters as parameters
from branch_and_price import BranchAndPrice
from data_classes import NodeLabel, CGDualVariablesFromMaster
from l_shaped_algorithm import LShapedAlgorithm



def run_monolithic_model():
    model = Model(sites.SITE_LIST)
    model.solve_and_print_model()

def run_b_and_p_gurobi():
    bp = BranchAndPrice()
    bp.branch_and_price(l_shaped=False)


def run_b_and_p_l_shaped():
    bp = BranchAndPrice()
    bp.branch_and_price(l_shaped=True)






if __name__ == '__main__':
    if configs.ALGORITHM == 0:
        run_b_and_p_l_shaped()

    elif configs.ALGORITHM == 1:
        run_b_and_p_gurobi()

    else:
        run_monolithic_model()


