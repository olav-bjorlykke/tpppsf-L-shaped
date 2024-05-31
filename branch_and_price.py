from cg_master_problem import CGMasterProblem, CGDualVariablesFromMaster
import initialization.sites as sites
from model import Model
from data_classes import NodeLabel
import initialization.configs as configs
import copy
from gurobipy import GRB # type: ignore
import logging
from l_shaped_algorithm import LShapedAlgorithm
from time import perf_counter
import multiprocessing

class BranchAndPrice:
    def __init__(self):
        self.master = CGMasterProblem()
        self.set_up_logging()
        self.upper_bound = 1000000000000
        self.lower_bound = 0

    def branch_and_price(self, l_shaped: bool):
        self.generate_initial_columns()
        #Initializing branch and price
        current_best_solution = 0
        root = NodeLabel(number=0, parent=0, level=0)
        q = [root]
        solved_nodes = []
        node_number = 0
        branched_indexes = sites.NODE_INIT_LIST
        for indexes in branched_indexes:
            root.up_branching_indices[indexes[0]].append(indexes[1])

        current_node = root
        while q:
            node_start = perf_counter()
            if q[0].level > current_node.level: self.upper_bound = self.get_upper_bounds(solved_nodes, current_node.level)
            current_node = q.pop(0)
            feasible = False
            if l_shaped:
                feasible = self.column_generation_ls(current_node)
            else:
                feasible = self.column_generation(current_node)

            self.master_logger.info(f"Solved node: {current_node.number}")
            if not feasible: #Pruning criteria 1
                #Prunes the node if the Node is not feasible
                solved_nodes.append(current_node)
                continue
            lp_solution = self.master.model.getObjective().getValue()
            integer_feasible = self.master.check_integer_feasible()
            branching_variable = self.master.get_branching_variable(current_node)
            self.master.model.setParam('OutputFlag', 1)
            self.master.update_and_solve_as_mip(current_node)
            if self.master.model.status != GRB.OPTIMAL:
                self.master.model.computeIIS()
                self.master.model.write("master_mip_model.ilp")
            else:
                mip_solution = self.master.model.getObjective().getValue()
            if lp_solution < self.lower_bound: #Pruning criteria 2
                #Prunes the node if the LP solution is worse than the current best MIP solution found
                current_node.LP_solution = lp_solution
                current_node.MIP_solution = mip_solution
                solved_nodes.append(current_node)
                node_end = perf_counter()
                self.bp_logger.info(f"PRUNED - Node: {current_node.number} / iteration {self.master.iterations_k} / up_branching: {current_node.up_branching_indices} / down_branching: {current_node.down_branching_indices} / parent: {current_node.parent} / time in node : {node_end - node_start}")
                continue
            if integer_feasible: #Pruning criteria 3
                #If the LP solution is integer feasible -> prune!
                current_best_solution = lp_solution
                current_node.MIP_solution = lp_solution
                current_node.integer_feasible = True
                solved_nodes.append(current_node)
                node_end = perf_counter()
                if lp_solution > self.lower_bound:
                    self.lower_bound = lp_solution
                self.bp_logger.info(f"PRUNED - Node: {current_node.number} / iteration {self.master.iterations_k} / up_branching: {current_node.up_branching_indices} / down_branching: {current_node.down_branching_indices} / parent: {current_node.parent} / lp_solution:{current_node.LP_solution} / mip_solution:{current_node.MIP_solution} / level:{current_node.level} / ub:{self.upper_bound} / lb:{self.lower_bound} / gap:{100 * (self.upper_bound - self.lower_bound)/self.upper_bound}% / time in node : {node_end - node_start}")
                continue

            #Setting the variables in the NodeLabel object
            current_node.LP_solution = lp_solution
            current_node.MIP_solution = mip_solution
            if mip_solution > self.lower_bound: self.lower_bound = mip_solution


            mip_solution = 0
            lp_solution = 0
            #Appending the node to solved nodes
            solved_nodes.append(current_node)
            self.general_logger.info(solved_nodes)
            #Deciding which variable to branch on
                        # Returns a list with [location, index] for the branching variable
            branched_indexes.append(branching_variable)
            node_end = perf_counter()
            self.bp_logger.info(f"Node: {current_node.number} / iteration {self.master.iterations_k} / up_branching: {current_node.up_branching_indices} / down_branching: {current_node.down_branching_indices} / parent: {current_node.parent} / lp_solution:{current_node.LP_solution} / mip_solution:{current_node.MIP_solution} / level:{current_node.level} / ub:{self.upper_bound} / lb:{self.lower_bound} / gap:{100 * (self.upper_bound - self.lower_bound)/self.upper_bound}% / time in node : {node_end - node_start}")
            #Book keeping to store the branching index
            new_up_branching_indicies = copy.deepcopy(current_node.up_branching_indices)
            new_down_branching_indicies = copy.deepcopy(current_node.down_branching_indices)
            #Branching variable is a list with two elements - we append the second element to the correct location by indexing on the first element
            new_up_branching_indicies[branching_variable[0]].append(branching_variable[1])
            new_down_branching_indicies[branching_variable[0]].append(branching_variable[1])
            #Logging
            self.general_logger.info(f"BRANCH INDICE UP {new_up_branching_indicies}")
            self.general_logger.info(f"BRANCH INDICE DOWN {new_down_branching_indicies}")
            self.general_logger.info(f"Branched indexes:{branched_indexes}")
            #Creating child nodes
            node_number += 1
            new_node_up = NodeLabel(number=node_number, 
                                    parent=current_node.number, 
                                    level=current_node.level+1, 
                                    up_branching_indices=new_up_branching_indicies, 
                                    down_branching_indices=current_node.down_branching_indices)
            #Add one extra to the node_number to ensure they have different numbers
            node_number += 1
            new_node_down = NodeLabel(number=node_number, 
                                      parent=current_node.number, 
                                      level=current_node.level+1, 
                                      up_branching_indices=current_node.up_branching_indices, 
                                      down_branching_indices=new_down_branching_indicies)
            #Append new nodes to the search queue
            q.append(new_node_up)
            q.append(new_node_down)
            
            # optimality_gap = Current Best / Generation lowest LP

    def column_generation(self, node_label):
        #Initializing sub problems
        sub_problems = [Model(sites.SITE_LIST[i], self.master.iterations_k) for i in range(len(sites.SITE_LIST))]

        self.master.model.setParam('OutputFlag', 0)

        previous_dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        dual_variables = CGDualVariablesFromMaster()
        while previous_dual_variables != dual_variables or self.master.iterations_k < 8:
            previous_dual_variables = dual_variables
            for i, sub in enumerate(sub_problems):
                sub.solve_as_sub_problem(dual_variables, up_branching_indices=node_label.up_branching_indices[i],
                                         down_branching_indices=node_label.down_branching_indices[i],
                                         iteration=self.master.iterations_k)
                if sub.model.status != GRB.OPTIMAL:
                    self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE in sub_problem {i}")
                    return False

                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                #column.write_to_file()
                #TODO:mAKE CONDITIONAL ON sub-problem being feasible
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.model.objVal}")
            self.master.update_model(node_label) 
            self.master.solve()
            #self.master.model.write(f"master_{self.master.iterations_k}.lp")
            #self.master.model.write(f"master_model_iteration_{self.master.iterations_k}.lp")
            if self.master.model.status == GRB.INF_OR_UNBD:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE!")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")                    
            dual_variables = self.master.get_dual_variables()
            #dual_variables.write_to_file()

        return True

    def column_generation_ls(self, node_label):
        #Initializing sub problems
        self.master.model.setParam('OutputFlag', 0)

        previous_dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        dual_variables = CGDualVariablesFromMaster()
        sub_problems = [LShapedAlgorithm(sites.SITE_LIST[i], i, node_label) for i in range(len(sites.SITE_LIST))]
        
        while previous_dual_variables != dual_variables or self.master.iterations_k < 8:
            previous_dual_variables = dual_variables
            for i, sub in enumerate(sub_problems):
                self.general_logger.info(f"## solved subproblem {i}, iteration {self.master.iterations_k} ")
                feasible = sub.solve(dual_variables)
                if not feasible:
                    return False
                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                column.write_to_file()
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.master.model.objVal}") #TODO: this does not account for solving the subsubs as MIPs
                self.sub_logger.info(
                    f"Reduced cost of column iteration {self.master.iterations_k}, site{i}: {column.calculate_reduced_cost(dual_variables)}")
            self.master.update_model(node_label) 
            self.master.solve()                  
            if self.master.model.status != GRB.OPTIMAL:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: status: {self.master.model.status}")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")                    
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

        return True

    def column_generation_ls_parallel(self, node_label):
        # Initializing sub problems
        self.master.model.setParam('OutputFlag', 0)

        previous_dual_variables = CGDualVariablesFromMaster(u_EOH=[1 for _ in range(configs.NUM_SCENARIOS)])
        dual_variables = CGDualVariablesFromMaster()

        while previous_dual_variables != dual_variables or self.master.iterations_k < 8:
            previous_dual_variables = dual_variables
            new_columns = self.run_sub_problems_in_parallel(dual_variables, node_label)
            for column in new_columns:
                self.master.columns[(column.site, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {column.site}:{column.calculate_reduced_cost(dual_variables)}")  # TODO: this does not account for solving the subsubs as MIPs
            self.master.update_model(node_label)
            self.master.solve()
            if self.master.model.status != GRB.OPTIMAL:  # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: status: {self.master.model.status}")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

        return True
        pass

    def generate_initial_columns(self):
        initial = Model(sites.SITE_LIST)
        initial_columns = initial.create_zero_column(0)
        initial_columns2 = initial.create_initial_columns(1)


        for column in initial_columns:
            self.master.columns[(column.site, column.iteration_k)] = column


        for column in initial_columns2:
           self.master.columns[(column.site, column.iteration_k)] = column

        self.master.initialize_model()
        self.master.iterations_k = 2

    def get_upper_bounds(self, solved_nodes, level):
        lvl_upper_bound = 0
        for node in solved_nodes:
            if (node.level == level) and (node.LP_solution > lvl_upper_bound):
                lvl_upper_bound = node.LP_solution

        return lvl_upper_bound

    def run_sub_problems_in_parallel(self, dual_variables, node_label):
        processes = []
        column_queue = multiprocessing.Queue()
        feasible_queue = multiprocessing.Queue()

        for i, site in enumerate(sites.SITE_LIST):
            p = multiprocessing.Process(target=self.create_and_solve_sub_problem, args=(i, dual_variables, node_label, site, self.master.iterations_k, column_queue, feasible_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        columns = [column_queue.get() for _ in processes]
        return columns

    def create_and_solve_sub_problem(self, i, dual_variables,node_label, site, iterations_k, column_queue, feasible_queue):
        sub_problem = LShapedAlgorithm(site, i, node_label)
        feasible = sub_problem.solve(dual_variables)
        if not feasible:
            feasible_queue.put(feasible)
            return False
        column_queue.put(sub_problem.get_column_object(iterations_k))

    def set_up_logging(self):
        path = configs.LOG_DIR
        logging.basicConfig(
            level=logging.INFO,
            filemode='a'  # Set filemode to 'w' for writing (use 'a' to append)
        )
        self.general_logger = logging.getLogger(f"general_logger")

        #Creating logger for logging master problem values
        self.master_logger = logging.getLogger("master_logger")
        file_handler1 = logging.FileHandler(f'{path}master_logger.log')
        file_handler1.setLevel(logging.INFO)
        self.master_logger.addHandler(file_handler1)


        #Creating logger for logging sub-problem values
        self.sub_logger = logging.getLogger("sub_logger")
        file_handler2 = logging.FileHandler(f'{path}sub_logger.log')
        file_handler2.setLevel(logging.INFO)
        self.sub_logger.addHandler(file_handler2)

        #Creating logger for Branch and Price:
        self.bp_logger = logging.getLogger("bp_logger")
        file_handler3 = logging.FileHandler(f'{path}bp_logger.log')
        file_handler3.setLevel(logging.INFO)
        self.bp_logger.addHandler(file_handler3)




if __name__ == '__main__':
    bp = BranchAndPrice()
    bp.branch_and_price()

