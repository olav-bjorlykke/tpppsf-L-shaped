from l_shaped_master_problem import LShapedMasterProblem
from l_shaped_sub_problem import LShapedSubProblem
import initialization.configs as configs
import initialization.sites as sites
from initialization.input_data import InputData
from gurobipy import GRB


class LShapedAlgoritm:
    def __init__(self) -> None:
        self.input_data = InputData()


    def run(self):
        master_problem = LShapedMasterProblem(self.input_data, sites.short_sites_list[0], 0)   #TODO: Fix how this takes in the site to solve for
        master_problem.initialize_model()                                                                    #Create the gurobi model object within the master-problem class
        #Solve the master problem with no cuts
        master_problem.solve()
        iteration_counter = 1

        #Write initial objective value to file
        self.write_obj_value_to_file(master_problem, iteration=iteration_counter)

        # Sets the old master problem to be none in the first iteration
        old_master_problem_solution = None
        # new master problem solution set to be the solution with no cuts, this is an Lshaped data class object
        new_master_problem_solution = master_problem.get_variable_values()
        #Initializes a list of L-shaped sub-problems
        subproblems = [LShapedSubProblem(s, sites.short_sites_list[0], 0, new_master_problem_solution, self.input_data) for s in range(configs.NUM_SCENARIOS)] #TODO:Fix input of sites
        #Initializes an empyt list for dual variable tracking
        dual_variables = [None for _ in range(configs.NUM_SCENARIOS)]
        for s in range(configs.NUM_SCENARIOS):
            #Initializes the gurobi model for all sub-problems
            subproblems[s].initialize_model()
        while new_master_problem_solution != old_master_problem_solution:
            iteration_counter += 1
            #Sets the previous solution to be the solution found in the last iteration, before finding a new solution
            old_master_problem_solution = new_master_problem_solution
            #Solve the sub-problem for every scenario, with new fixed variables from master problem
            for s in range(configs.NUM_SCENARIOS):
                subproblems[s].update_model(new_master_problem_solution)
                subproblems[s].solve()
                #Fetch dual variables from sub-problem, and write to list so they can be passed to the master problem
                dual_variables[s] = subproblems[s].get_dual_values()
            #Add new optimality cut, based on dual variables
            master_problem.add_optimality_cuts(dual_variables)
            #Solve master problem with new cuts, and store the variable values to be passed to sub-problems in next iteration
            master_problem.solve()
            new_master_problem_solution = master_problem.get_variable_values()
            #Log the objective value
            self.write_obj_value_to_file(master_problem, iteration=iteration_counter)


        #Once the L-shaped terminates, we solve it as a MIP to generate an integer feasible solution
        for s in range(configs.NUM_SCENARIOS):
            subproblems[s].update_model_to_mip(new_master_problem_solution)
            subproblems[s].solve()

        #This prints the solution to file -> Can be deleted once integrated with colum generation
        new_master_problem_solution.write_to_file()
        for s in range(configs.NUM_SCENARIOS):
            subproblems[s].print_variable_values()


    def write_obj_value_to_file(self, master_problem, iteration):
        f = open(f"{configs.OUTPUT_DIR}L_shapedMP_obj_value.txt", "a")
        param_name = "ObjVal"
        f.write(f"MP solution iteration {iteration}: {master_problem.model.getAttr(param_name)}\n")
        f.close()


if __name__ == "__main__":
    LShapedAlgoritm().run()


