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
import initialization.sites as sites


def run_monolithic_model():
    model = Model(sites.short_sites_list[0])
    model.solve_and_print_model()
    column = model.get_column_object(1,1)
    column.write_to_file()
    print(column)

def main():
    pass

if __name__ == '__main__':
    run_monolithic_model()




