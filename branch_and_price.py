from cg_master_problem import CGMasterProblem, CGDualVariablesFromMaster
import initialization.sites as sites
from model import Model
from data_classes import NodeLabel
import initialization.configs as configs
import copy
from gurobipy import GRB # type: ignore
import logging

class BranchAndPrice:
    def __init__(self):
        self.master = CGMasterProblem()
        self.set_up_logging()

    def branch_and_price(self):
        self.generate_initial_columns()
        current_best_solution = 0
        root = NodeLabel(number=0, parent=0, level=0)
        q = [root]
        solved_nodes = []
        node_number = 0
        branched_indexes = sites.NODE_INIT_LIST

        while q:
            print(q)
            current_node = q.pop(0)
            feasible = self.column_generation_test(current_node)
            if not feasible: #Pruning criteria 1
                solved_nodes.append(current_node)
                continue
            solution = self.master.model.getObjective().getValue()
            if solution < current_best_solution: #Pruning criteria 2
               current_node.LP_solution = solution
               solved_nodes.append(current_node)
               continue
            integer_feasible = self.master.check_integer_feasible()
            if integer_feasible: #Pruning criteria 3
                current_best_solution = solution
                current_node.MIP_solution = solution
                solved_nodes.append(current_node)
                continue
            current_node.LP_solution = solution
            solved_nodes.append(current_node)
            print(solved_nodes)
            branching_variable = self.master.get_branching_variable(branched_indexes)              # Returns a list with [location, index] for the branching variable
            branched_indexes.append(branching_variable)
            self.bp_logger.info(f"Node: {current_node.number} / iteration {self.master.iterations_k} / branching on variable: {branching_variable}")
            node_number += 1
            new_up_branching_indicies = copy.deepcopy(current_node.up_branching_indices)
            new_down_branching_indicies = copy.deepcopy(current_node.down_branching_indices)
            new_up_branching_indicies[branching_variable[0]].append(branching_variable[1])
            new_down_branching_indicies[branching_variable[0]].append(branching_variable[1])
            print("BRANCH INDICE UP",new_up_branching_indicies)
            print("BRANCH INDICE DOWN",new_down_branching_indicies)
            new_node_up = NodeLabel(number=node_number, 
                                    parent=current_node.number, 
                                    level=current_node.level+1, 
                                    up_branching_indices=new_up_branching_indicies, 
                                    down_branching_indices=current_node.down_branching_indices)
            node_number += 1
            new_node_down = NodeLabel(number=node_number, 
                                      parent=current_node.number, 
                                      level=current_node.level+1, 
                                      up_branching_indices=current_node.up_branching_indices, 
                                      down_branching_indices=new_down_branching_indicies)
            q.append(new_node_up)
            q.append(new_node_down)
            
            # optimality_gap = Current Best / Generation lowest LP


    def column_generation(self, node_label):
        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], self.master.iterations_k) for i in range(len(sites.SITE_LIST))]

        self.master.model.setParam('OutputFlag', 0)

        previous_dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        dual_variables = CGDualVariablesFromMaster()
        while previous_dual_variables != dual_variables:
            previous_dual_variables = dual_variables
            for i, sub in enumerate(sub_problems):
                sub.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[i],
                                         down_branching_indices=node_label.down_branching_indices[i],
                                         iteration=self.master.iterations_k)

                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                column.write_to_file()
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.model.objVal}")
            self.master.update_model(node_label) 
            self.master.solve()                  
            if self.master.model.status != GRB.OPTIMAL:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE!")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")                    
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

        return True


    def column_generation_test(self, node_label):
        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], self.master.iterations_k) for i in range(len(sites.SITE_LIST))]
        self.master.model.setParam('OutputFlag', 0)
        dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        for _ in range(5):
            
            for i, sub in enumerate(sub_problems):
                sub.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[i],
                                         down_branching_indices=node_label.down_branching_indices[i],
                                         iteration=self.master.iterations_k)

                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                column.write_to_file()
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.model.objVal}")
            self.master.update_model(node_label) 
            self.master.solve()                  #TODO:I think maybe this becomes infeasible when introducing the branching constraints and solving before generating more columns - any way to get around this?
            if self.master.model.status != GRB.OPTIMAL:  # To prevent errors, handled by pruning in B&P
                self.master.iterations_k -= 1
                self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE!") 
                return False
            self.master_logger.info(f"{self.master.iterations_k}:objective = {self.master.model.objVal}")                    
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

        return True



    def generate_initial_columns(self):
        initial = Model(sites.SITE_LIST)
        initial_columns = initial.create_initial_columns(0)
        initial_columns2 = initial.create_zero_column(1)


        for column in initial_columns:
            self.master.columns[(column.site, column.iteration_k)] = column


        for column in initial_columns2:
            self.master.columns[(column.site, column.iteration_k)] = column

        self.master.initialize_model()
        self.master.iterations_k += 1

    def set_up_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            filemode='a'  # Set filemode to 'w' for writing (use 'a' to append)
        )
        self.general_logger = logging.getLogger("general_logger")

        #Creating logger for logging master problem values
        self.master_logger = logging.getLogger("master_logger")
        file_handler1 = logging.FileHandler('master_logger.log')
        file_handler1.setLevel(logging.INFO)
        self.master_logger.addHandler(file_handler1)


        #Creating logger for logging sub-problem values
        self.sub_logger = logging.getLogger("sub_logger")
        file_handler2 = logging.FileHandler('sub_logger.log')
        file_handler2.setLevel(logging.INFO)
        self.sub_logger.addHandler(file_handler2)

        #Creating logger for Branch and Price:
        self.bp_logger = logging.getLogger("bp_logger")
        file_handler3 = logging.FileHandler('bp_logger.log')
        file_handler3.setLevel(logging.INFO)
        self.bp_logger.addHandler(file_handler3)




    """
    def write_node_to_file(self, node_label):
        f = open(f"{configs.OUTPUT_DIR}Branch_and_Price_nodes.txt", "a")
        f.write(f"Number: {node_label.number}, 
                level: {node_label.level}, 
                parent: {node_label.parent}, 
                LP-solution: {node_label.LP_solution}, 
                MIP-solution: {node_label.MIP_solution}, 
                Up-indicies: {node_label.up_branching_indices}, 
                Down-indicies: {node_label.down_branching_indices} \n")
        f.close()
    """

if __name__ == '__main__':
    bp = BranchAndPrice()
    bp.branch_and_price()

