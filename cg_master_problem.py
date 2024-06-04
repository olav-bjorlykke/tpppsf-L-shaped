import gurobipy as gp
from gurobipy import GRB 
import initialization.parameters as parameters
from data_classes import CGDualVariablesFromMaster, NodeLabel


class CGMasterProblem:
    def __init__(self, configs):
        self.iterations_k = 1
        self.configs = configs
        self.l_size = self.configs.NUM_LOCATIONS
        self.f_size = self.configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.s_size = self.configs.NUM_SCENARIOS
        self.columns = {} #dict, key = (site, iteration), value = CGcolumn object

    """
    Initializing and updating model
    """
    def initialize_model(self):
        self.model = gp.Model(f"CG master problem model")
        self.declare_variables()
        self.set_objective()
        self.add_MAB_company_constraint()
        self.add_convexity_constraint()
        self.add_EOH_constraint()
        self.add_variable_tracking_constraints()

    def update_model(self, node_label):
        self.iterations_k += 1
        print(self.iterations_k)
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
        self.lambda_var = self.model.addVars(self.configs.NUM_LOCATIONS, self.iterations_k, vtype=GRB.CONTINUOUS, lb=0, name="lambda_var")
        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1,name="deploy_bin")

    def declare_mip_variables(self):
        self.lambda_var = self.model.addVars(self.configs.NUM_LOCATIONS, self.iterations_k, vtype=GRB.BINARY, lb=0, name="lambda_var")
        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="deploy_bin")

    def solve(self):
        self.model.optimize()

    def update_and_solve_as_mip(self, node_label):
        self.model.remove(self.model.getConstrs())
        self.model.remove(self.model.getVars())
        # Stopping the model after 1800 seconds
        self.model.setParam("TimeLimit", 1800)
        self.declare_mip_variables()
        self.set_objective()
        self.add_MAB_company_constraint()
        self.add_convexity_constraint()
        self.add_EOH_constraint()
        self.add_variable_tracking_constraints()
        self.add_branching_constraints(node_label)
        self.model.optimize()
        #Resetting the time limit so as not to interfere with other instances
        self.model.setParam("TimeLimit", 200000)

    """
    Add objective
    """

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(
                gp.quicksum(
                    self.configs.SCENARIO_PROBABILITIES[s] *
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
            ) <= self.configs.MAB_COMPANY_LIMIT * 1.000001 #The 1.001 factor is here to deal with slight numeric instability when exporting variables from the sub-problem
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
            ) >= self.configs.MAB_COMPANY_LIMIT * parameters.EOH_ratio_requirement * 0.99999 #The 0.9999 factor is here to deal with slight numeric instability when exporting variables from the sub-problem
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
                ), name="UP-Branch"
                )
        
        for location in range(len(node_label.down_branching_indices)):
            for indicie in node_label.down_branching_indices[location]:
                self.model.addConstr((
                    self.deploy_bin[location, indicie] == 0
                ), name="DOWN-Branch"
                )

    """
    Variable tracking constraints
    """
    def add_variable_tracking_constraints(self):
        self.model.addConstrs((
            gp.quicksum(
                self.lambda_var[l, k] * self.columns[(l, k)].production_schedules[t].deploy_bin[t] if t in self.columns[(l,k)].production_schedules.keys() else 0.0
                for k in range(self.iterations_k)
            ) == self.deploy_bin[l, t]
            for t in range(self.t_size)
            for l in range(self.l_size)
        ), name=f"deploy_bin_tracking")


    def get_dual_variables(self):
        dual_variables = CGDualVariablesFromMaster(self.configs)
        dual_variables.iteration = self.iterations_k
        for t in range(self.t_size + 1):
            for s in range(self.s_size):
                constr = self.model.getConstrByName(f"MAB[{t},{s}]")
                dual = constr.getAttr("Pi")
                dual_variables.u_MAB[t][s] = dual

        for s in range(self.s_size):
            constr = self.model.getConstrByName(f"EOH[{s}]")
            dual = constr.getAttr("Pi")
            dual_variables.u_EOH[s] = dual

        for l in range(self.l_size):
            constr = self.model.getConstrByName(f"Convexity[{l}]")
            dual = constr.getAttr("Pi")
            dual_variables.v_l[l] = dual
        return dual_variables

    def get_branching_variable(self, node_label: NodeLabel):
        closest_to_1_location_and_index = [0, 0]
        value_closest_to_1 = 0
        for l in range(self.l_size):
            for t in range(self.t_size):
                if (self.deploy_bin[l, t].X > value_closest_to_1) and ((t not in node_label.up_branching_indices[l]) and (t not in node_label.down_branching_indices[l])):
                    value_closest_to_1 = self.deploy_bin[l, t].X
                    closest_to_1_location_and_index = [l, t]
        return closest_to_1_location_and_index

    def check_integer_feasible(self):
        for i in range(self.configs.NUM_LOCATIONS):
            for j in range(self.iterations_k):
                if not (self.lambda_var[i,j].x == 0 or self.lambda_var[i,j].x == 1):
                    return False
        return True        
        
