import time

from data_classes import NodeLabel
from model import Model
from initialization.configs import Configs
from initialization.input_data import InputData
from initialization.sites import Sites
from branch_and_price import BranchAndPrice
from initialization.sites import Sites
from l_shaped_algorithm import LShapedAlgorithm
from data_classes import CGDualVariablesFromMaster



def run_monolithic_model(configs):
    sites = Sites(configs)
    model = Model(sites.SITE_LIST, configs)
    model.solve_and_print_model()

def run_b_and_p_gurobi(configs, input_data):
    bp = BranchAndPrice(configs, input_data)
    bp.branch_and_price(l_shaped=False)


def run_b_and_p_l_shaped(configs, input_data):
    bp = BranchAndPrice(configs, input_data)
    bp.branch_and_price(l_shaped=True)

def run_ls_single_site(site, site_index, configs, node_label, input_data, cg_dual_variables):
    ls = LShapedAlgorithm(site, site_index, configs, node_label, input_data)
    start = time.perf_counter()
    ls.solve(cg_dual_variables)
    column = ls.get_column_object(iteration=1)
    column.write_to_file()
    end = time.perf_counter()
    ls.ls_logger.info(f"####### Solved the L-SHAPED problem in {end - start} seconds #############")

def run_gb_single_site(configs):
    sites = Sites(configs)
    model = Model(sites.SITE_LIST, configs)
    model.solve_as_single_site_mip()


def main():
    configs = Configs()
    input_data = InputData(configs)
    if configs.INSTANCE == "SINGLE_SITE":
        if configs.ALGORITHM == 0:
            sites = Sites(configs)
            node_label = NodeLabel(configs)
            branched_indexes = sites.NODE_INIT_LIST
            for indexes in branched_indexes:
                node_label.up_branching_indices[indexes[0]].append(indexes[1])
            cg_dual_variables = CGDualVariablesFromMaster(configs)
            run_ls_single_site(sites.SITE_LIST[0], 0, configs, node_label, input_data, cg_dual_variables)
        if configs.ALGORITHM == 2:
            run_gb_single_site(configs)

    else:
        if configs.ALGORITHM == 0:
            run_b_and_p_l_shaped(configs, input_data)

        elif configs.ALGORITHM == 1:
            run_b_and_p_gurobi(configs, input_data)

        else:
            run_monolithic_model(configs)

def single_site_main():
    configs = Configs()
    input_data = InputData(configs)
    sites = Sites(configs)
    node_label = NodeLabel(configs)
    branched_indexes = sites.NODE_INIT_LIST
    for indexes in branched_indexes:
        node_label.up_branching_indices[indexes[0]].append(indexes[1])
    cg_dual_variables = CGDualVariablesFromMaster(configs)

    instance = input("Input 'ls' for LSHAPED, input 'm' for monolithic model: ")
    if instance == "ls":
        run_ls_single_site(sites.SITE_LIST[0], 0, configs, node_label, input_data, cg_dual_variables)

    elif instance == "m":
        run_gb_single_site(0, configs)

    else:
        print("Chosen instance does not exist")


if __name__ == '__main__':
    main()



