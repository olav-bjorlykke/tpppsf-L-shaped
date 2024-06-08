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
import logging


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
    ls.solve_with_parallelization(cg_dual_variables)
    column = ls.get_column_object(iteration=1)
    column.write_to_file()
    end = time.perf_counter()
    ls.ls_logger.info(f"####### Solved the L-SHAPED problem in {end - start} seconds #############")

def run_gb_single_site(configs):
    sites = Sites(configs)
    model = Model(sites.SITE_LIST, configs)
    model.solve_as_single_site_mip()

def run_sensitivity_analysis():
    configs = Configs()
    logger = set_up_logging(configs)
    input_data = InputData(configs)
    sites = Sites(configs)
    model = Model(sites.SITE_LIST, configs)
    model.solve_and_print_model()
    columns = model.get_columns_from_multisite_solution(0)
    for i in range(100):
        instance = configs.INSTANCE
        algorithm = configs.ALGORITHM
        configs = Configs(scenarios=1, instance=instance, algorithm=algorithm, random_scenearios=True)
        new_sites = Sites(configs)
        model = Model(new_sites.SITE_LIST, configs)
        model.solve_and_print_model()
        optimal_objective = model.model.objVal
        model.solve_with_locked_first_stage_vars(columns, i)
        locked_objective = model.model.objVal
        locked_model_status = model.model.status
        logger.info(f"Iteration {i}: optimal objective = {optimal_objective}, locked objective = {locked_objective}, difference absolute = {optimal_objective-locked_objective}, difference_percentage = {(optimal_objective - locked_objective)/optimal_objective},  status = {locked_model_status}")

def set_up_logging(configs):
    path = configs.LOG_DIR
    logging.basicConfig(
        level=logging.INFO,
        filemode='a'  # Set filemode to 'w' for writing (use 'a' to append)
    )
    general_logger = logging.getLogger(f"general_logger")
    file_handler = logging.FileHandler(f'{path}logger.log')
    file_handler.setLevel(logging.INFO)
    general_logger.addHandler(file_handler)
    return general_logger

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


if __name__ == '__main__':
    run_sensitivity_analysis()



