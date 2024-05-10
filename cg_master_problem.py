import gurobipy as gp
from gurobipy import GRB
import initialization.parameters as parameters
import initialization.configs as configs



class CGMasterProblem:
    def __init__(self):
        self.iterations_k = 0
        self.l_size = configs.NUM_LOCATIONS
        self.f_size = configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.s_size = configs.NUM_SCENARIOS
        self.columns = {} #dict, key = [site, iteration], value = CGcolumn object

    """
    Initializing and updating model
    """
    def initialize_model(self): #TODO:FIx after constraints are made
        self.model = gp.Model(f"CG master problem model")
        self.declare_variables()
        self.add_MAB_company_constraint()
        self.add_convexity_constraint()
        self.add_EOH_constraint()
        self.add_variable_tracking_constraints()
        self.set_objective()

    def update_model(self):
        pass

    def declare_variables(self):
        self.lambda_var = self.model.addVars(configs.NUM_LOCATIONS, self.iterations_k, vtype=GRB.CONTINUOUS, lb=0)
        
        # Declaring the tracking variables
        self.y = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0)
        self.x = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size+1, self.s_size, vtype=GRB.CONTINUOUS, lb=0)
        self.w = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0)

        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.deploy_type_bin = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.employ_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.employ_bin_granular = self.model.addVars(self.l_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.harvest_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)

    def solve(self):
        self.model.optimize()

    """
    Add objective
    """

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(
                    gp.quicksum(
                        parameters.scenario_probabilities[s] *
                        gp.quicksum(
                            self.columns[[l, k]].production_schedules[t_hat].w[s][t][f]
                            for t_hat in list(self.columns[[l, k]].production_schedules.keys())
                            for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, parameters.number_periods))
                            for f in range(self.f_size)
                        )
                        for s in range(self.s_size)
                    )
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
        self.model.addConstrs(
            gp.quicksum(
                gp.quicksum(
                    self.columns[[l, k]].production_schedules[t_hat].x[f][t][s]
                    for t_hat in list(self.columns[[l, k]].production_schedules.keys())
                    for f in range(self.f_size)
                )
                * self.lambda_var[l, k]
                for l in range(self.l_size)
                for k in range(self.iterations_k)
            ) <= configs.MAB_COMPANY_LIMIT
            for t in range(self.t_size)
            for s in range(self.s_size)
        )

    def add_EOH_constraint(self):
        self.model.addConstrs(
            gp.quicksum(
                gp.quicksum(
                    self.columns[[l, k]].production_schedules[t_hat].x[f][parameters.number_periods - 1][s]
                    for t_hat in list(self.columns[[l, k]].production_schedules.keys()) if t_hat >= self.t_size - parameters.max_periods_deployed #TODO: Can remove if statement, but this is a tighter formulation
                    for f in range(self.f_size)
                )
                * self.lambda_var[l, k]
                for l in range(self.l_size)
                for k in range(self.iterations_k)
            ) <= configs.MAB_COMPANY_LIMIT * parameters.EOH_ratio_requirement
            for s in range(self.s_size)
        )

    def add_convexity_constraint(self):
        self.model.addConstrs(
            gp.quicksum(
                self.lambda_var[l,k] for k in range(self.iterations_k)
            ) == 1
            for l in range(self.l_size)
        )



    """
    Branching constraints
    """
    def add_branching_constraints(self):
        #TODO:Add once format of branching information is set
        pass

    """
    Variable tracking constraints
    """
    def add_variable_tracking_constraints(self):
        self.model.addConstrs(
            gp.quicksum(
                    self.lambda_var[l,k] * self.columns[[l, k]].production_schedules[t_hat].w[f][t][s] if t_hat in self.columns[l,k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.w[l,f,t_hat,t,s]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(self.t_size)
            for s in range(self.s_size)
        )

        self.model.addConstrs(
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].x[f][t][s] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.x[l, f, t_hat, t, s]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(self.t_size + 1)
            for s in range(self.s_size)
        )

        self.model.addConstrs(
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].y[f][t] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.y[l, f, t]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
        )

        self.model.addConstrs(
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].deploy_bin[t] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.deploy_bin[l, t]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
        )

        self.model.addConstrs(
            gp.quicksum(gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].deploy_type_bin[f][t] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.deploy_type_bin[l, f, t]
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
            )
        )

        self.model.addConstrs(
            gp.quicksum(gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].employ_bin[t][s] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.employ_bin[l, t, s]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
            for s in range(self.s_size)
            )
        )

        self.model.addConstrs(
            gp.quicksum(gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].employ_bin_granular[t][s] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.employ_bin_granular[l, t_hat, t, s]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
            for s in range(self.s_size)
            )
        )

        self.model.addConstrs(
            gp.quicksum(gp.quicksum(
                self.lambda_var[l, k] * self.columns[[l, k]].production_schedules[t_hat].employ_bin_granular[t][s] if t_hat in self.columns[l, k].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.employ_bin_granular[l, t, s]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, self.t_size)
            for s in range(self.s_size)
            )
        )


    #TODO: Get shadow prices out!

