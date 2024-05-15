import gurobipy as gp
from gurobipy import GRB
import initialization.parameters as parameters
import initialization.configs as configs
from data_classes import CGDualVariablesFromMaster



class CGMasterProblem:
    def __init__(self):
        self.iterations_k = 1
        self.l_size = configs.NUM_LOCATIONS
        self.f_size = configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.s_size = configs.NUM_SCENARIOS
        self.columns = {} #dict, key = (site, iteration), value = CGcolumn object

    """
    Initializing and updating model
    """
    def initialize_model(self): #TODO:FIx after constraints are made
        self.model = gp.Model(f"CG master problem model")
        self.declare_variables()
        self.set_objective()
        self.add_MAB_company_constraint()
        self.add_convexity_constraint()
        self.add_EOH_constraint()
        self.add_variable_tracking_constraints()


    def update_model(self, node_label):
        self.iterations_k += 1
        self.model.remove(self.model.getConstrs())
        self.model.remove(self.model.getVars())
        self.declare_variables()
        self.set_objective()
        self.add_MAB_company_constraint()
        self.add_convexity_constraint()
        self.add_EOH_constraint()
        self.add_variable_tracking_constraints()
        self.add_branching_constraints(node_label)


    def declare_variables(self):
        self.lambda_var = self.model.addVars(configs.NUM_LOCATIONS, self.iterations_k, vtype=GRB.CONTINUOUS, lb=0, name="lambda_var")
        
        # Declaring the tracking variables
        self.y = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="Y")
        self.x = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size+1, self.s_size, vtype=GRB.CONTINUOUS, lb=0, name="X")
        self.w = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0,name="W")

        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="deploy_bin")
        self.deploy_type_bin = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="deploy_type")
        self.employ_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="employ_bin")
        self.employ_bin_granular = self.model.addVars(self.l_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="employ_bin_gran")
        self.harvest_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="harvest_bin")

    def solve(self):
        self.model.optimize()

    """
    Add objective
    """

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(
                    configs.SCENARIO_PROBABILITIES[s] *
                    gp.quicksum(
                        self.columns[(l, k)].production_schedules[t_hat].w[f][t][s]
                        for t_hat in list(self.columns[(l, k)].production_schedules.keys())
                        for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, parameters.number_periods))
                        for f in range(self.f_size)
                    )
                    for s in range(self.s_size)
                ) * self.lambda_var[l, k]
                for l in range(self.l_size)
                for k in range(self.iterations_k)
            ),
            GRB.MAXIMIZE
        )



    """
    Add constraints
    """

    def add_MAB_company_constraint(self):
        self.model.addConstrs((
            gp.quicksum(
                gp.quicksum(
                    self.columns[(l, k)].production_schedules[t_hat].x[f][t][s]
                    for t_hat in list(self.columns[(l, k)].production_schedules.keys())
                    for f in range(self.f_size)
                )
                * self.lambda_var[l, k]
                for l in range(self.l_size)
                for k in range(self.iterations_k)
            ) <= configs.MAB_COMPANY_LIMIT * 1.001 #The 1.001 factor is here to deal with slight numeric instability when exporting variables from the sub-problem
            for t in range(self.t_size + 1)
            for s in range(self.s_size)
        ), name="MAB"
        )

    def add_EOH_constraint(self):
        self.model.addConstrs((
            gp.quicksum(
                gp.quicksum(
                    self.columns[(l, k)].production_schedules[t_hat].x[f][parameters.number_periods][s]
                    for t_hat in list(self.columns[(l, k)].production_schedules.keys())
                    for f in range(self.f_size)
                )
                * self.lambda_var[l, k]
                for l in range(self.l_size)
                for k in range(self.iterations_k)
            ) >= configs.MAB_COMPANY_LIMIT * parameters.EOH_ratio_requirement * 0.9999 #The 0.9999 factor is here to deal with slight numeric instability when exporting variables from the sub-problem
            for s in range(self.s_size)
        ), name="EOH"
        )

    def add_convexity_constraint(self):
        self.model.addConstrs((
            gp.quicksum(
                self.lambda_var[l,k] for k in range(self.iterations_k)
            ) == 1
            for l in range(self.l_size)
        ), name="Convexity"
        )



    """
    Branching constraints
    """
    def add_branching_constraints(self, node_label):
        for location in range(len(node_label.up_branching_indices)):
            for indicie in node_label.up_branching_indices[location]:
                self.model.addConstr((
                    self.deploy_bin[location, indicie] == 1
                ))
        
        for location in range(len(node_label.down_branching_indices)):
            for indicie in node_label.down_branching_indices[location]:
                self.model.addConstr((
                    self.deploy_bin[location, indicie] == 0
                ))
    

            

    """
    Variable tracking constraints
    """
    def add_variable_tracking_constraints(self):
        self.model.addConstrs((
            gp.quicksum(
                    self.lambda_var[l,k] * self.columns[(l, k)].production_schedules[t_hat].w[f][t][s] if t_hat in self.columns[(l,k)].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.w[l,f,t_hat,t,s]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(self.t_size)
            for s in range(self.s_size)
        ), name="W-tracking"
        )

        self.model.addConstrs((
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].x[f][t][s] if t_hat in self.columns[(l, k)].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.x[l, f, t_hat, t, s]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(self.t_size + 1)
            for s in range(self.s_size)
        ), name="X-tracking")

        added_y_tracking_indices = [] #Just introduced a fuckload of tech-debt
        for t_hat in range(self.t_size):
            for l in range(self.l_size):
                for k_hat in range(self.iterations_k):
                    if t_hat in self.columns[(l, k_hat)].production_schedules.keys():
                        added_y_tracking_indices.append((l,k_hat,t_hat))
                        if (l,k_hat,t_hat) not in added_y_tracking_indices:
                            self.model.addConstrs((
                                gp.quicksum(
                                    self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].y[f][t]
                                    for k in range(self.iterations_k)
                                ) == self.y[l, f, t]
                                for f in range(self.f_size)
                                for t in range(t_hat, self.t_size)
                            ), name="Y-tracking")

        added_deploy_bin_tracking_indices = []
        for t_hat in range(self.t_size):
            for l in range(self.l_size):
                for k_hat in range(self.iterations_k):
                    added_deploy_bin_tracking_indices.append((l, k_hat, t_hat))
                    if (l, k_hat, t_hat) not in added_deploy_bin_tracking_indices:
                        self.model.addConstrs((
                            gp.quicksum(
                                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].deploy_bin[t]
                                for k in range(self.iterations_k)
                            ) == self.deploy_bin[l, t]
                            for t in range(t_hat, self.t_size)
                        ), name="Deploy_bin-tracking")

        added_deploy_type_bin_tracking_indices = []
        for t_hat in range(self.t_size):
            for l in range(self.l_size):
                for k_hat in range(self.iterations_k):
                    added_deploy_type_bin_tracking_indices.append((l, k_hat, t_hat))
                    if (l, k_hat, t_hat) not in added_deploy_type_bin_tracking_indices:
                        self.model.addConstrs((
                            gp.quicksum(
                                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].deploy_type_bin[f][t]
                                for k in range(self.iterations_k)
                            ) == self.deploy_type_bin[l, f, t]
                            for f in range(self.f_size)
                            for t in range(t_hat, self.t_size)
                        ), name="deploy_type_bin-tracking")

        added_employ_bin_tracking_indices = []
        for t_hat in range(self.t_size):
            for l in range(self.l_size):
                for k_hat in range(self.iterations_k):
                    added_employ_bin_tracking_indices.append((l, k_hat, t_hat))
                    if (l, k_hat, t_hat) not in added_employ_bin_tracking_indices:
                        self.model.addConstrs((
                            gp.quicksum(
                                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].employ_bin[t][s]
                                for k in range(self.iterations_k)
                            ) == self.employ_bin[l, t, s]
                            for t in range(t_hat, self.t_size)
                            for s in range(self.s_size)
                        ), name="employ_bin-tracking")

        self.model.addConstrs((
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].employ_bin_granular[t][s] if t_hat in self.columns[(l, k)].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.employ_bin_granular[l, t_hat, t, s]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
            for s in range(self.s_size)

        ), name="employ_bin_granular-tracking")

        added_harvest_bin_tracking_indices = []
        for t_hat in range(self.t_size):
            for l in range(self.l_size):
                for k_hat in range(self.iterations_k):
                    added_harvest_bin_tracking_indices.append((l, k_hat, t_hat))
                    if (l, k_hat, t_hat) not in added_harvest_bin_tracking_indices:
                        self.model.addConstrs((
                            gp.quicksum(
                                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t_hat].harvest_bin[t][s]
                                for k in range(self.iterations_k)
                            ) == self.harvest_bin[l, t, s]
                            for t in range(t_hat, self.t_size)
                            for s in range(self.s_size)

                        ), name="Harvest_bin-tracking")

    def get_dual_variables(self):
        dual_variables = CGDualVariablesFromMaster(iteration=self.iterations_k)
        for t in range(self.t_size + 1):
            for s in range(self.s_size):
                constr = self.model.getConstrByName(f"MAB[{t},{s}]")
                dual = constr.getAttr("Pi")
                dual_variables.u_MAB[t][s] = dual

        for s in range(self.s_size):
            constr = self.model.getConstrByName(f"EOH[{s}]")
            dual = constr.getAttr("Pi")
            dual_variables.u_EOH[s] = dual

        return dual_variables

    def get_branching_variable(self):
        closest_to_1_location_and_index = (0, 0)
        value_closest_to_1 = 0
        for l in range(self.l_size):
            for t in range(self.t_size):
                if self.deploy_bin[l, t].X > value_closest_to_1 and self.deploy_bin[l,t].X != 1:
                    value_closest_to_1 = self.deploy_bin[l,t].X
                    closest_to_1_location_and_index = l, t
        return closest_to_1_location_and_index

    def check_integer_feasible(self):
        for i in range(configs.NUM_LOCATIONS):
            for j in range(self.iterations_k):
                if not (self.lambda_var[i,j].x == 0 or self.lambda_var[i,j].x == 1):
                    return False
        return True        
        
