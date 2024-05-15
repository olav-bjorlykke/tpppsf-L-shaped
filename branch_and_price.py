from cg_master_problem import CGMasterProblem
import initialization.sites as sites
from model import Model
from data_classes import NodeLabel

class BranchAndPrice:
    def __init__(self):
        pass

    def branch_and_price(self):
        self.master = CGMasterProblem()
        q = []
        solved_nodes = []
        #que = [Root Node]
        # while que.not_empty:
            #current_node = q.pop()
            #Solve column generation in current node
            #If dominating MIP solution found -> Prune current node
            #Else if: Solution is Integer Feasible and better than current best -> update current best
            #Else: Find branching variable -> Generate Child Nodes -> Add child nodes to que
            # optimality_gap = Current Best / Generation lowest LP

    def column_generation(self, node_label: NodeLabel, master):
        optimal = False

        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], master.iterations_k + 1) for i in range(len(sites.SITE_LIST))]

        while not optimal:
            master.update_model()
            master.solve()
            dual_variables = master.get_dual_variables()
            dual_variables.write_to_file()

            for l in range(3):
                for k in range(master.iterations_k):
                    print(master.lambda_var[l, k].x)


            columns_previously_generated = True
            for i, sub in enumerate(sub_problems):
                sub.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[i],
                                         down_branching_indices=node_label.down_branching_indices[i],
                                         iteration=master.iterations_k)

                column = sub.get_column_object(iteration=master.iterations_k)
                column.site = i
                column.write_to_file()

                if column not in master.columns.values():
                    columns_previously_generated = False

                master.columns[(i, master.iterations_k)] = column

            optimal = columns_previously_generated
        print(f"########  Solved node {NodeLabel.number} ############")



    def get_branching_variable(self):
        pass





    def generate_initial_columns(self):
        pass

