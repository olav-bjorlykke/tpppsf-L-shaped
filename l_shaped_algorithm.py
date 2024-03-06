from l_shaped_master_problem import LShapedMasterProblem
from l_shaped_sub_problem import LShapedSubProblem
import initialization.configs as configs
import initialization.sites as sites
from gurobipy import GRB

class LShapedAlgoritm:
    def __init__(self) -> None:
        pass


    def run():
        master_problem = LShapedMasterProblem(sites.short_sites_list, 0) # TODO: change input
        master_problem.initialize_model()
        master_problem.solve()
        old_master_problem_solution = None
        new_master_problem_solution = master_problem.get_variable_values()
        subproblems = [LShapedSubProblem(s, 0, new_master_problem_solution, sites.short_sites_list) for s in range(configs.NUM_SCENARIOS)]
        dual_variables = [None for _ in range(configs.NUM_SCENARIOS)]
        for s in range(configs.NUM_SCENARIOS):
                subproblems[s].initialize_model()
        while new_master_problem_solution != old_master_problem_solution:
            old_master_problem_solution = new_master_problem_solution
            for s in range(configs.NUM_SCENARIOS):
                subproblems[s].update_model(new_master_problem_solution)
                subproblems[s].solve()
                dual_variables[s] = subproblems[s].get_dual_values()
            master_problem.add_optimality_cuts(dual_variables)
            master_problem.solve()
            new_master_problem_solution = master_problem.get_variable_values()
        return master_problem.model.getAttr("ObjVal")

    run()