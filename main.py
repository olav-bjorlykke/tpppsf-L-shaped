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


def main():
    pass

if __name__ == '__main__':
    model = Model(sites.SITE_LIST)
    model.solve_and_print_model()
