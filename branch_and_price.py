from cg_master_problem import CGMasterProblem, CGDualVariablesFromMaster
import initialization.sites as sites
from model import Model
from data_classes import NodeLabel
import initialization.configs as configs
from gurobipy import GRB

class BranchAndPrice:
    def __init__(self):
        self.master = CGMasterProblem()

    def branch_and_price(self):
        self.generate_initial_columns()
        current_best_solution = 0
        root = NodeLabel(number=0, parent=0, level=0)
        q = [root]
        solved_nodes = []
        node_number = 0

        while q:
            current_node = q.pop(0)
            feasible = self.column_generation(current_node)
            solved_nodes.append(current_node)
            if not feasible: #Pruning criteria 1
                continue
            solution = self.master.model.getObjective().getValue()
            if solution < current_best_solution: #Pruning criteria 2
               current_node.LP_solution = solution
               continue
            integer_feasible = self.master.check_integer_feasible()
            if integer_feasible: #Pruning criteria 3
                current_best_solution = solution
                current_node.MIP_solution = solution
                continue
            location, index = self.master.get_branching_variable()
            node_number += 1
            new_up_branching_indicies = current_node.up_branching_indices
            new_down_branching_indicies = current_node.down_branching_indices
            new_up_branching_indicies[location].append(index)
            new_down_branching_indicies[location].append(index)
            new_node_up = NodeLabel(number=node_number, parent=current_node.number, level=current_node.level+1, up_branching_indices=new_up_branching_indicies, down_branching_indices=current_node.down_branching_indices)
            node_number += 1
            new_node_down = NodeLabel(number=node_number, parent=current_node.number, level=current_node.level+1, up_branching_indices=current_node.up_branching_indices, down_branching_indices=new_down_branching_indicies)
            q.append(new_node_up)
            q.append(new_node_down)
            
            # optimality_gap = Current Best / Generation lowest LP




    def column_generation(self, node_label):

        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], self.master.iterations_k + 1) for i in range(len(sites.SITE_LIST))]
        
        previous_dual_variables = CGDualVariablesFromMaster()
        dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        while previous_dual_variables != dual_variables:
            previous_dual_variables = dual_variables
            self.master.update_model(node_label) # TODO: NEED TO IMPLEMENT BRANCHING CONSTRAINTS IN THE MASTER AND SUB-PROBLEMS
            self.master.solve()
            if self.master.model.status == GRB.INFEASIBLE:
                return False                     # To prevent errors, handled by pruning in B&P
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

            for l in range(3):
                for k in range(self.master.iterations_k):
                    print(self.master.lambda_var[l, k].x)

            for i, sub in enumerate(sub_problems):
                sub.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[i],
                                         down_branching_indices=node_label.down_branching_indices[i],
                                         iteration=self.master.iterations_k)

                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                column.write_to_file()
                self.master.columns[(i, self.master.iterations_k)] = column

            
        print(f"########  Solved node {node_label.number} ############")
        return True


    def generate_initial_columns(self):
        initial = Model(sites.SITE_LIST)
        initial_columns = initial.create_initial_columns(0)
        initial_columns2 = initial.create_zero_column(1)


        for column in initial_columns:
            self.master.columns[(column.site, column.iteration_k)] = column
            #column.write_to_file()

        for column in initial_columns2:
            self.master.columns[(column.site, column.iteration_k)] = column
            #column.write_to_file()

        self.master.initialize_model()

if __name__ == '__main__':
    bp = BranchAndPrice()
    bp.branch_and_price()

