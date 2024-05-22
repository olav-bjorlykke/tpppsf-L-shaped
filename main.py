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





if __name__ == '__main__':
    site = sites.SITE_LIST[0]
    node_label = NodeLabel(0, 0, 0)
    #node_label.up_branching_indices[0].append(7)
    dual_variables = CGDualVariablesFromMaster()
    dual_variables.u_MAB[13][0] = 2.331633
    dual_variables.u_MAB[12][1] = 5.921953
    dual_variables.u_MAB[35][0] = 2.75
    dual_variables.u_MAB[38][1] = 3.2499
    dual_variables.u_MAB[45][0] = 0.2
    dual_variables.u_MAB[46][1] = 0.2
    dual_variables.u_MAB[35][0] = 2.75
    dual_variables.u_MAB[58][1] = 0.1
    dual_variables.u_MAB[3][0] = 2.75
    dual_variables.u_MAB[5][1] = 5.1

    algo = LShapedAlgorithm(site, 0, node_label=node_label)
    algo.solve(dual_variables)
    column = algo.get_column_object(iteration=1)
    column.write_to_file()

    mon_model = Model(site)
    mon_model.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[0])
    column2 = mon_model.get_column_object(iteration=100)
    column2.write_to_file()



