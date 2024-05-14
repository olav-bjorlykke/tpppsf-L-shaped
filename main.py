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


def run_monolithic_model():
    model = Model(sites.SITE_LIST)
    model.solve_and_print_model()
    model.model.write("model.lp")
    columns = model.create_zero_column(0)
    for column in columns:
        column.write_to_file()

def create_zero_column():
    initial = Model(sites.SITE_LIST)
    initial_columns = initial.create_zero_column(0)
    initial.model.write("zero.lp")
    if initial.model.status == GRB.OPTIMAL:
        # Open a file to write variable values
        with open("variable_values.txt", "w") as file:
            # Iterate through variables and write their values to the file
            for var in initial.model.getVars():
                file.write(f"{var.varName}: {var.x}\n")
        print("Variable values written to variable_values.txt")
    else:
        print("Optimization was not successful")


    for column in initial_columns:
        column.write_to_file()

def column_generation():
    master = CGMasterProblem()
    initial = Model(sites.SITE_LIST)
    initial_columns = initial.create_initial_columns(0)
    initial_columns2 = initial.create_zero_column(1)

    for column in initial_columns:
        master.columns[(column.site, column.iteration_k)] = column
        column.write_to_file()

    for column in initial_columns2:
        master.columns[(column.site, column.iteration_k)] = column
        column.write_to_file()

    master.initialize_model()
    optimal = False
    j = 2
    while not optimal:
        master.update_model(iteration=j)
        master.solve()
        if master.model.status != GRB.OPTIMAL:
            master.model.computeIIS()
            master.model.write("model.ilp")
        else:
            master.model.write(f"model{j}.lp")
            for l in range(3):
                for k in range(j):
                    print(master.lambda_var[l, k].x)
        dual_variables = master.get_dual_variables()
        dual_variables.write_to_file()

        for i in range(len(sites.SITE_LIST)):
            print(i, sites.SITE_LIST[i].name)
            sub = Model(sites.SITE_LIST[i], iterations=j)
            sub.solve_as_sub_problem(dual_variables)
            sub.model.write(f"sub_model_site{i}_iter{j}.lp")
            column = sub.get_column_object(iteration=j)
            column.site = i
            column.write_to_file()
            print("Adding colum to master:", i, j)
            master.columns[(i, j)] = column
        print(master.columns.keys())

        sum = 0.0
        for l in range(configs.NUM_LOCATIONS):
            sum += master.lambda_var[l, j - 1].x
        if sum == 0.0 and j >= 4:
            optimal = True
        j += 1




def main():
    pass

if __name__ == '__main__':
    #run_monolithic_model()
    column_generation()
    #create_zero_column()



