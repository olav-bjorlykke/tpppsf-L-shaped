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


def run_monolithic_model():
    model = Model(sites.short_sites_list)
    columns = model.create_zero_column(0)
    for column in columns:
        column.write_to_file()


def run_master_problem():
    master = CGMasterProblem()
    initial = Model(sites.SITE_LIST)
    initial_columns = initial.create_zero_column(0)
    initial_columns2 = initial.create_initial_columns(1)

    for column in initial_columns:
        master.columns[(column.site, column.iteration_k)] = column

    for column in initial_columns2:
        master.columns[(column.site, column.iteration_k)] = column

    for j in range(2,10):
        master.initialize_model()
        master.solve()
        dual_variables = master.get_dual_variables()
        dual_variables.write_to_file()

        columns = []
        for i in range(len(sites.SITE_LIST)):
            sub = Model(sites.SITE_LIST[i])
            sub.solve_as_sub_problem(dual_variables)
            column = sub.get_column_object(iteration=j)
            columns.append(column)

        for column in columns:
            master.columns[(column.site, column.iteration_k)] = column





def main():
    pass

if __name__ == '__main__':
    run_master_problem()




