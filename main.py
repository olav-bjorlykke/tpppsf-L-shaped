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
    model = Model(sites.short_sites_list[0])
    model.solve_and_print_model()
    column = model.get_column_object(1,1)
    column.write_to_file()
    print(column)

def run_master_problem():
    master = CGMasterProblem()
    sub = Model(sites.short_sites_list[0])
    sub.solve_and_print_model()
    column = sub.get_column_object(location=0,iteration=0)
    column.write_to_file()
    master.columns[(0,0)] = column
    master.initialize_model()
    master.solve()
    master.model.write("master.lp")




def main():
    pass

if __name__ == '__main__':
    run_master_problem()




