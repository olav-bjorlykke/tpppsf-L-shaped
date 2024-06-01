from l_shaped_master_problem import LShapedMasterProblem
from l_shaped_sub_problem import LShapedSubProblem
from data_classes import CGColumn, DeployPeriodVariables
from gurobipy import GRB
import logging

class LShapedAlgorithm:
    def __init__(self, site, site_index, configs, node_label, input_data) -> None:
        self.configs = configs
        self.master = LShapedMasterProblem(site, site_index, self.configs, input_data)
        self.site = site
        self.l = site_index
        self.node_label = node_label
        self.subproblems = []
        self.set_up_logging()

    def solve(self, cg_dual_variables):
        self.master.initialize_model(self.node_label, cg_dual_variables)                                                                    #Create the gurobi model object within the master-problem class
        #Solve the master problem with no cuts
        self.master.solve()
        iteration_counter = 1

        if self.master.model.status != GRB.OPTIMAL:
            self.master.model.computeIIS()
            self.master.model.write("L-shaped-master-problem.ilp")
            return False

        # Sets the old master problem to be none in the first iteration
        old_master_problem_solution = None
        # new master problem solution set to be the solution with no cuts, this is an Lshaped data class object
        new_master_problem_solution = self.master.get_variable_values()
        #Initializes a list of L-shaped sub-problems
        subproblems = [LShapedSubProblem(scenario=s, site=self.site,site_index= self.l,fixed_variables= new_master_problem_solution, cg_dual_variables=cg_dual_variables, configs=self.configs) for s in range(self.configs.NUM_SCENARIOS)]
        #Initializes an empyt list for dual variable tracking
        dual_variables = [None for _ in range(self.configs.NUM_SCENARIOS)]
        for s in range(self.configs.NUM_SCENARIOS):
            #Initializes the gurobi model for all sub-problems
            subproblems[s].initialize_model()
        while new_master_problem_solution != old_master_problem_solution:
            iteration_counter += 1
            #Sets the previous solution to be the solution found in the last iteration, before finding a new solution
            old_master_problem_solution = new_master_problem_solution
            #Solve the sub-problem for every scenario, with new fixed variables from master problem
            for s in range(self.configs.NUM_SCENARIOS):
                subproblems[s].update_model(new_master_problem_solution)
                subproblems[s].solve()
                if subproblems[s].model.status != GRB.OPTIMAL:
                    subproblems[s].model.computeIIS()
                    subproblems[s].model.write(f"subsub{s}.ilp")
                self.ls_logger.info(f"{iteration_counter} Sub: {s}: Objective: {subproblems[s].model.objVal}")
                #Fetch dual variables from sub-problem, and write to list so they can be passed to the master problem
                dual_variables[s] = subproblems[s].get_dual_values()
            #Add new optimality cut, based on dual variables
            self.master.add_optimality_cuts(dual_variables)
            #Solve master problem with new cuts, and store the variable values to be passed to sub-problems in next iteration
            self.master.solve()
            self.ls_logger.info(f"{iteration_counter} master:{self.master.model.objVal}")
            new_master_problem_solution = self.master.get_variable_values()

        #Once the L-shaped terminates, we solve it as a MIP to generate an integer feasible solution
        for s in range(self.configs.NUM_SCENARIOS):
            subproblems[s].update_model_to_mip(new_master_problem_solution)
            subproblems[s].solve()
            if subproblems[s].model.status != GRB.OPTIMAL:
                subproblems[s].model.computeIIS()
                subproblems[s].model.write(f"subsub{s}_mip.ilp")
                subproblems[s].model.write(f"subsub{s}_mip.lp")
                return False
        self.subproblems = subproblems
        return True


    def get_column_object(self, iteration):
        column = CGColumn(self.l, iteration)
        for t_hat in self.master.get_deploy_period_list():
            deploy_period_variables = DeployPeriodVariables(self.configs)
            for f in range(self.master.f_size):
                for t in range(self.master.t_size):
                    deploy_period_variables.y[f][t] = round(self.master.y[f, t].x, 5)
                    deploy_period_variables.deploy_type_bin[f][t] = round(self.master.deploy_type_bin[f, t].x, 5)
                    for s, sub in enumerate(self.subproblems):
                        deploy_period_variables.w[f][t][s] = round(sub.w[f, t_hat, t].x, 5)
                for t in range(self.master.t_size +1):
                    for s, sub in enumerate(self.subproblems):
                        deploy_period_variables.x[f][t][s] = round(sub.x[f,t_hat,t].x, 5)
            for t in range(self.master.t_size):
                deploy_period_variables.deploy_bin[t] = round(self.master.deploy_bin[t].x,5)
                for s, sub in enumerate(self.subproblems):
                    deploy_period_variables.employ_bin[t][s] = round(sub.employ_bin[t].x,5)
                    deploy_period_variables.employ_bin_granular[t][s] = round(sub.employ_bin_granular[t_hat, t].x,5)
                    deploy_period_variables.harvest_bin[t][s] = round(sub.harvest_bin[t].x,5)
            column.production_schedules[t_hat] = deploy_period_variables
        return column

    def set_up_logging(self):
        path = self.configs.LOG_DIR
        logging.basicConfig(
            level=logging.INFO,
            filemode='a'  # Set filemode to 'w' for writing (use 'a' to append)
        )
        # Creating logger for logging master problem values
        self.ls_logger = logging.getLogger("ls_logger")
        file_handler1 = logging.FileHandler(f'{path}ls_logger.log')
        file_handler1.setLevel(logging.INFO)
        self.ls_logger.addHandler(file_handler1)





