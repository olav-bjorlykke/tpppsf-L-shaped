from l_shaped_master_problem import LShapedMasterProblem
from l_shaped_sub_problem import LShapedSubProblem
import initialization.configs as configs
import initialization.sites as sites
from initialization.input_data import InputData


class LShapedAlgoritm:
    def __init__(self) -> None:
        self.input_data = InputData()


    def run(self):

        master_problem = LShapedMasterProblem(self.input_data, sites.short_sites_list[0], 0)
        master_problem.initialize_model()
        master_problem.solve()
        master_problem.model.write("model_0.rlp")
        f = open("test_results_2.txt", "a")
        f.write(f"MP solution iteration 1: {master_problem.model.getAttr("ObjVal")}\n")
        iteration_counter = 1
        f.close()
        old_master_problem_solution = None
        new_master_problem_solution = master_problem.get_variable_values()
        subproblems = [LShapedSubProblem(s, sites.short_sites_list[0], 0, new_master_problem_solution, self.input_data) for s in range(configs.NUM_SCENARIOS)]
        dual_variables = [None for _ in range(configs.NUM_SCENARIOS)]
        for s in range(configs.NUM_SCENARIOS):
                subproblems[s].initialize_model()
        while new_master_problem_solution != old_master_problem_solution:
            iteration_counter += 1
            old_master_problem_solution = new_master_problem_solution
            for s in range(configs.NUM_SCENARIOS):
                subproblems[s].update_model(new_master_problem_solution)
                subproblems[s].solve()
                dual_variables[s] = subproblems[s].get_dual_values()
            master_problem.add_optimality_cuts(dual_variables)
            master_problem.solve()
            f = open("test_results_2.txt", "a")
            f.write(f"MP solution iteration {iteration_counter}: {master_problem.model.getAttr("ObjVal")}\n")
            f.close()
            new_master_problem_solution = master_problem.get_variable_values()
        return master_problem.model.getAttr("ObjVal")

LShapedAlgoritm().run()

