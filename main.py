from model import Model
from initialization.configs import Configs
from initialization.input_data import InputData
from initialization.sites import Sites
from branch_and_price import BranchAndPrice



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


if __name__ == '__main__':
    configs = Configs()
    input_data = InputData(configs)
    if configs.ALGORITHM == 0:
        run_b_and_p_l_shaped(configs, input_data)

    elif configs.ALGORITHM == 1:
        run_b_and_p_gurobi(configs, input_data)

    else:
        run_monolithic_model(configs)


