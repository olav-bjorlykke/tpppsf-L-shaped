import time

from cg_master_problem import CGMasterProblem, CGDualVariablesFromMaster
from initialization.sites import Sites
from model import Model
from data_classes import NodeLabel
import copy
from gurobipy import GRB
import logging
from l_shaped_algorithm import LShapedAlgorithm
from time import perf_counter
import multiprocessing

class BranchAndPrice:
    def __init__(self, configs, input_data):
        self.configs = configs
        self.input_data = input_data
        self.master = CGMasterProblem(self.configs)
        self.set_up_logging()
        self.upper_bound = 1000000000000
        self.lower_bound = 0
        self.sites = Sites(configs)
        
    def branch_and_price(self, l_shaped: bool):
        self.generate_initial_columns()
        #Initializing branch and price
        root = NodeLabel(configs=self.configs)
        q = [root]
        solved_nodes = []
        node_number = 0
        branched_indexes = self.sites.NODE_INIT_LIST
        for indexes in branched_indexes:
            root.up_branching_indices[indexes[0]].append(indexes[1])

        current_node = root
        while q:
            node_start = perf_counter()
            if q[0].level > current_node.level: self.upper_bound = self.get_upper_bounds(solved_nodes, current_node.level)
            current_node = q.pop(0)
            feasible = False
            if l_shaped:
                feasible = self.column_generation_ls_parallel(current_node)
            else:
                feasible = self.column_generation_parallel(current_node)

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
            new_node_up = NodeLabel(configs=self.configs) 
            new_node_up.number = node_number
            new_node_up.parent = current_node.number 
            new_node_up.level = current_node.level+1 
            new_node_up.up_branching_indices=new_up_branching_indicies 
            new_node_up.down_branching_indices=current_node.down_branching_indices
            #Add one extra to the node_number to ensure they have different numbers
            node_number += 1
            new_node_down = NodeLabel(configs=self.configs)
            new_node_down.number=node_number
            new_node_down.parent=current_node.number
            new_node_down.level=current_node.level+1 
            new_node_down.up_branching_indices=current_node.up_branching_indices
            new_node_down.down_branching_indices=new_down_branching_indicies
            #Append new nodes to the search queue
            q.append(new_node_up)
            q.append(new_node_down)
            
    def column_generation(self, node_label):
        #Initializing sub problems
        sub_problems = [Model(self.sites.SITE_LIST[i], self.configs, iterations=self.master.iterations_k) for i in range(len(self.sites.SITE_LIST))]
        self.master.model.setParam('OutputFlag', 0)
        previous_dual_variables = CGDualVariablesFromMaster(self.configs)
        previous_dual_variables.u_MAB[0][0] = 10
        dual_variables = CGDualVariablesFromMaster(self.configs)
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
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.model.objVal}")
            self.master.update_model(node_label) 
            self.master.solve()
            if self.master.model.status == GRB.INF_OR_UNBD:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE!")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")                    
            dual_variables = self.master.get_dual_variables()
        return True
    
    def column_generation_parallel(self, node_label):
        self.master.model.setParam('OutputFlag', 0)
        previous_dual_variables = CGDualVariablesFromMaster(self.configs)
        previous_dual_variables.u_MAB[0][0] = 10
        dual_variables = CGDualVariablesFromMaster(self.configs)
        prev_objective = 100000000000
        objective = 99999999
        iterations = 0
        while (previous_dual_variables != dual_variables and (objective - prev_objective > 0.1)) or iterations < 3:
            iterations += 1
            prev_objective = objective
            previous_dual_variables = dual_variables
            sub_problems_start = time.perf_counter()
            new_columns = self.cg_run_sub_problems_in_parallel(dual_variables, node_label)
            sub_problems_end = time.perf_counter()
            for column in new_columns:
                self.master.columns[(column.site, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {column.site}: {column.calculate_reduced_cost(self.configs, dual_variables)}")
            self.master.update_model(node_label)
            master_start_time = time.perf_counter()
            self.master.solve()
            master_end_time = time.perf_counter()
            if self.master.model.status == GRB.INF_OR_UNBD:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: INFEASIBLE!")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal} / time in master: {master_end_time - master_start_time} / time in subs: {sub_problems_end - sub_problems_start} ")
            objective = self.master.model.objVal
            dual_variables = self.master.get_dual_variables()
        return True

    def column_generation_ls(self, node_label):
        #Initializing sub problems
        self.master.model.setParam('OutputFlag', 0)

        previous_dual_variables = CGDualVariablesFromMaster(self.configs)
        previous_dual_variables.u_MAB[0][0] = 10
        dual_variables = CGDualVariablesFromMaster(self.configs)
        sub_problems = [LShapedAlgorithm(self.sites.SITE_LIST[i], i, self.configs, node_label, self.input_data) for i in range(len(self.sites.SITE_LIST))]
        prev_objective = 100000000000
        objective = 99999999
        iterations = 0
        while (previous_dual_variables != dual_variables and (objective - prev_objective > 1)) or iterations < 5:
            iterations += 1
            prev_objective = objective
            previous_dual_variables = dual_variables
            for i, sub in enumerate(sub_problems):
                self.general_logger.info(f"## solved subproblem {i}, iteration {self.master.iterations_k} ")
                feasible = sub.solve(dual_variables)
                if not feasible:
                    return False
                column = sub.get_column_object(iteration=self.master.iterations_k)
                column.site = i
                column.write_to_file(self.configs)
                self.master.columns[(i, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {i}:{sub.master.model.objVal}") #TODO: this does not account for solving the subsubs as MIPs
                self.sub_logger.info(
                    f"Reduced cost of column iteration {self.master.iterations_k}, site{i}: {column.calculate_reduced_cost(self.configs, dual_variables)}")
            self.master.update_model(node_label) 
            self.master.solve()                  
            if self.master.model.status != GRB.OPTIMAL:     # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: status: {self.master.model.status}")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(f"{self.master.iterations_k}: objective = {self.master.model.objVal}")
            objective = self.master.model.objVal
            dual_variables = self.master.get_dual_variables()
            dual_variables.write_to_file()

        return True

    def column_generation_ls_parallel(self, node_label):
        self.master.model.setParam('OutputFlag', 0)
        previous_dual_variables = CGDualVariablesFromMaster(self.configs)
        previous_dual_variables.u_MAB[0][0] = 10
        dual_variables = CGDualVariablesFromMaster(self.configs)
        prev_objective = 100000000000
        objective = 999999999999
        iterations = 0
        while (previous_dual_variables != dual_variables and (objective - prev_objective > 1)) or iterations < 3:
            iterations += 1
            prev_objective = objective
            previous_dual_variables = dual_variables
            sub_problems_start = time.perf_counter()
            new_columns = self.run_sub_problems_in_parallel(dual_variables, node_label)
            sub_problems_end = time.perf_counter()
            for column in new_columns:
                self.master.columns[(column.site, self.master.iterations_k)] = column
                self.sub_logger.info(f"iteration {self.master.iterations_k} / site {column.site}: {column.calculate_reduced_cost(self.configs, dual_variables)}")  # TODO: this does not account for solving the subsubs as MIPs
            self.master.update_model(node_label)
            master_start_time = time.perf_counter()
            self.master.solve()
            master_end_time = time.perf_counter()
            if self.master.model.status != GRB.OPTIMAL:  # To prevent errors, handled by pruning in B&P
                self.master_logger.info(f"{self.master.iterations_k}: status: {self.master.model.status}")
                self.master.iterations_k -= 1
                return False
            self.master_logger.info(
                f"{self.master.iterations_k}: objective = {self.master.model.objVal} / time in master: {master_end_time - master_start_time} / time in subs: {sub_problems_end - sub_problems_start} ")
            dual_variables = self.master.get_dual_variables()
            objective = self.master.model.objVal
            dual_variables.write_to_file()
        return True

    def generate_initial_columns(self):
        initial = Model(self.sites.SITE_LIST, self.configs)
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

    def cg_run_sub_problems_in_parallel(self, dual_variables, node_label):
        processes = []
        column_queue = multiprocessing.Queue()
        feasible_queue = multiprocessing.Queue()
        for i, site in enumerate(self.sites.SITE_LIST):
            p = multiprocessing.Process(target=cg_create_and_solve_sub_problem, args=(i, dual_variables, node_label, site, self.master.iterations_k, column_queue, feasible_queue, self.configs))
            processes.append(p)
            p.start()
        columns = [column_queue.get() for _ in processes]
        for p in processes:
            p.join()
        return columns

    def run_sub_problems_in_parallel(self, dual_variables, node_label):
        processes = []
        column_queue = multiprocessing.Queue()
        feasible_queue = multiprocessing.Queue()
        for i, site in enumerate(self.sites.SITE_LIST):
            p = multiprocessing.Process(target=create_and_solve_sub_problem, args=(i, dual_variables, node_label, site, self.master.iterations_k, column_queue, feasible_queue, self.configs, self.input_data))
            processes.append(p)
            p.start()
        columns = [column_queue.get() for _ in processes]
        for p in processes:
            p.join()
        return columns
    
    def set_up_logging(self):
        path = self.configs.LOG_DIR
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

def create_and_solve_sub_problem(i, dual_variables,node_label, site, iterations_k, column_queue, feasible_queue, configs, input_data):
    sub_problem = LShapedAlgorithm(site, i, configs, node_label, input_data)
    feasible = sub_problem.solve(dual_variables)
    if not feasible:
        feasible_queue.put(feasible)
        return False
    column_queue.put(sub_problem.get_column_object(iterations_k))

def cg_create_and_solve_sub_problem(i, dual_variables,node_label, site, iterations_k, column_queue, feasible_queue, configs):
    sub_problem = Model(site, configs, iterations_k)
    feasible = sub_problem.solve_as_sub_problem(dual_variables, node_label.up_branching_indices[i], node_label.down_branching_indices[i], iteration=iterations_k, location=i)
    if not feasible:
        feasible_queue.put(feasible)
        return False
    column = sub_problem.get_column_object(iteration=iterations_k)
    column.site = i
    column_queue.put(column)

