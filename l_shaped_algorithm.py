from l_shaped_master_problem import LShapedMasterProblem
import initialization.sites as sites 

class LShapedAlgoritm:
    def __init__(self) -> None:
        pass


    def run():
        master_problem = LShapedMasterProblem(sites.short_sites_list, 0) # TODO: change input
        master_problem.initialize_model()
        master_problem.solve()
        old_master_problem_solution = None
        new_master_problem_solution = master_problem.get_variable_values()
        while new_master_problem_solution != old_master_problem_solution:
            old_master_problem_solution = new_master_problem_solution # Might need deepcopy here (?)
            # TODO: pass new_master_problem_solution to subproblems
            # TODO: solve subproblems and store dualvariables in a list of LShapedSubProblemDualVariable-objects, with dimension 1*s
            master_problem.add_optimality_cuts(dual_variables)
            master_problem.solve()
            new_master_problem_solution = master_problem.get_variable_values()
       #TODO: decide what to return. Will change when we integrate into DW-decomp

