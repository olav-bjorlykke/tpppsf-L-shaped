from cg_master_problem import CGMasterProblem
import initialization.sites as sites
from model import Model

class BranchAndPrice:
    def __init__(self):
        pass

    def branch_and_price(self):
        self.master = CGMasterProblem()
        #que = [Root Node]
        # while que.not_empty:
            #current_node = que.pop()
            #Solve column generation in current node
            #If dominating MIP solution found -> Prune current node
            #Else if: Solution is Integer Feasible and better than current best -> update current best
            #Else: Find branching variable -> Generate Child Nodes -> Add child nodes to que
            # optimality_gap = Current Best / Generation lowest LP

    def column_generation(self, node_label, master):
        optimal = False

        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], master.iterations_k + 1) for i in range(len(sites.SITE_LIST))]

        while not optimal:
            master.update_model()
            master.solve()
            dual_variables = master.get_dual_variables()

            iteration_columns = []
            for i, sub in enumerate(sub_problems):
                #TODO:implement branching in the sub-problems
                sub.solve_as_sub_problem(dual_variables)
                column = sub.get_column_object(master.iterations_k)
                column.site = i
                column.write_to_file()
                master.columns[(i, master.iterations_k)] = column
                iteration_columns.append(column)

            #TODO: Check if this block of code works
            columns_previously_generated = True
            for column in iteration_columns:
                if column not in master.columns.values():
                    columns_previously_generated = False

            optimal = columns_previously_generated


    def get_branching_variable(self):
        pass





    def generate_initial_columns(self):
        pass