from l_shaped_master_problem import LShapedMasterProblem
from l_shaped_sub_problem import LShapedSubProblem
import initialization.configs as configs
import initialization.sites as sites
from initialization.input_data import InputData


class LShapedAlgoritm:
    def __init__(self) -> None:
        self.input_data = InputData()


    def run(self):
        master_problem = LShapedMasterProblem(self.input_data, sites.short_sites_list[0], 0)        #Creates master problem object
        master_problem.initialize_model()                                                                    #Create the gurobi model object within the master-problem class
        master_problem.solve()                                                                               #Solve the model with no cuts


        # Code for writing to file
        master_problem.model.write("model_0.rlp")
        f = open("test_results_2.txt", "a")
        param_name = "ObjVal"
        f.write(f"MP solution iteration 1: {master_problem.model.getAttr(param_name)}\n")
        f.close()

        iteration_counter = 1

        old_master_problem_solution = None                                                                   #Sets the old master problem to be none in the first iteration
        new_master_problem_solution = master_problem.get_variable_values()                                   #new master problem set to be the solution from line 16
        #Initializes a list of L-shaped sub-problems
        subproblems = [LShapedSubProblem(s, sites.short_sites_list[0], 0, new_master_problem_solution, self.input_data) for s in range(configs.NUM_SCENARIOS)]
        #Initializes an empyt list for dual variable tracking
        dual_variables = [None for _ in range(configs.NUM_SCENARIOS)]
        for s in range(configs.NUM_SCENARIOS):
            #Initializes the gurobi model for all sub-problems
            subproblems[s].initialize_model()
        while new_master_problem_solution != old_master_problem_solution: #TODO: Check how this operator actually works
            iteration_counter += 1
            old_master_problem_solution = new_master_problem_solution
            for s in range(configs.NUM_SCENARIOS):
                subproblems[s].update_model(new_master_problem_solution)
                subproblems[s].solve()
                dual_variables[s] = subproblems[s].get_dual_values() #TODO:Check if this is fucking with the memory
            master_problem.add_optimality_cuts(dual_variables)
            master_problem.solve()

            f = open("test_results_2.txt", "a")
            f.write(f"MP solution iteration {iteration_counter}: {master_problem.model.getAttr(param_name)}\n")
            f.close()


            new_master_problem_solution = master_problem.get_variable_values()
        new_master_problem_solution.print()
        for s in range(configs.NUM_SCENARIOS):
            subproblems[s].print_variable_values()
        return master_problem.model.getAttr("ObjVal")


if __name__ == "__main__":
    LShapedAlgoritm().run()


