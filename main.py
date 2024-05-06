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
    sites.SITE_LIST[0].growth_sets.to_excel("growth_sets.xlsx")
