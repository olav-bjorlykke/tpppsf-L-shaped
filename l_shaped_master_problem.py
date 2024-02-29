import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from initialization.input_data import InputData
import initialization.parameters as parameters
import initialization.configs as configs
import initialization.sites as sites
from model import Model
from data_classes import LShapedMasterProblemVariables

class LShapedMasterProblem(Model):
    def __init__(self, site_objects, site_index,
                 MAB_shadow_prices_df=pd.DataFrame(), 
                 EOH_shadow_prices_df=pd.DataFrame(), 
                 input_data=InputData(), 
                 parameters=parameters, 
                 scenario_probabilities=configs.SCENARIO_PROBABILITIES):
        self.s_size = configs.NUM_SCENARIOS
        self.t_size = parameters.number_periods
        self.l = site_index
        super().__init__(site_objects, MAB_shadow_prices_df, EOH_shadow_prices_df, input_data, parameters, scenario_probabilities)


    def initialize_model(self):
        self.model = gp.Model(f"L-shaped master problem model")
        self.declare_variables()
        self.set_objective()
        self.add_initial_condition_constraint()
        self.add_smolt_deployment_constraints()
    
    def solve(self):
        self.model.optimize()
    
    def declare_variables(self):
        self.theta = self.model.addVars(self.s_size, vtype=GRB.CONTINUOUS, lb=0, name="theta")
        self.y = self.model.addVars(1, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.deploy_bin = self.model.addVars(1, self.t_size, vtype=GRB.BINARY, name="gamma") # 1 if smolt of any type is deplyed in t NOTE: No l-index as master problem for each l
        self.deploy_type_bin = self.model.addVars(1, self.f_size, self.t_size, vtype=GRB.BINARY, name="delta") # 1 if smolt type f is deployed in t NOTE: No l-index as master problem for each l 

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(
                self.scenario_probabilities[s] * self.theta[s] for s in range(self.s_size)
            )
            ,GRB.MAXIMIZE
        )
    
    def add_initial_condition_constraint(self): 
        if self.sites[self.l].init_biomass > 1:
            self.model.addConstr(
                self.y[self.l,0,0] == self.sites[self.l].init_biomass,
                name="Initial Condition"
            )
            self.model.addConstr(
                self.deploy_bin[self.l,0] == 1,
                name="Initial Condition bin"
            )

    def add_smolt_deployment_constraints(self):
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
            gp.quicksum(self.y[self.l, f, t] for f in range(self.f_size)) <= self.parameters.smolt_deployment_upper_bound * self.deploy_bin[self.l, t]
            # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
            for t in range(1, self.t_size)
        )
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
            gp.quicksum(self.y[self.l, f, t] for f in range(self.f_size)) >= self.parameters.smolt_deployment_lower_bound * self.deploy_bin[self.l,t]
            for t in range(1, self.t_size)
        )
        self.model.addConstrs(  # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
            self.y[self.l, f, t] >= self.parameters.smolt_deployment_lower_bound * self.deploy_type_bin[self.l,f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
        )
        self.model.addConstrs(
            # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
            self.parameters.smolt_deployment_upper_bound * self.deploy_type_bin[self.l, f, t] >=  self.y[self.l,f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
        )
        if self.sites[self.l].init_biomass < 1:
            self.model.addConstr(
                # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
                gp.quicksum(self.y[self.l, f, 0] for f in range(self.f_size)) <= self.parameters.smolt_deployment_upper_bound * self.deploy_bin[self.l, 0]
                ,name="Deploy limit if no init biomass at site"
            )
            self.model.addConstr(
                # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
                gp.quicksum(self.y[self.l, f, 0] for f in range(self.f_size)) >= self.parameters.smolt_deployment_lower_bound * self.deploy_bin[self.l, 0]
                , name="Deploy limit if no init biomass at site"
            )
            self.model.addConstrs(
                # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
                self.y[self.l, f, 0] >= self.parameters.smolt_deployment_lower_bound * self.deploy_type_bin[self.l, f, 0]
                for f in range(self.f_size)
            )
            self.model.addConstrs(
                # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
                self.parameters.smolt_deployment_upper_bound * self.deploy_type_bin[self.l, f, 0] >= self.y[self.l, f, 0]
                for f in range(self.f_size)
            )

    def add_optimality_cuts(self, dual_variables): # TODO: Implement with actual rho values based on datastructure generated from sub-problem
        self.model.addConstrs(
            self.theta[s] <= (
                gp.quicksum(dual_variables[s].rho_1[t] * (1 - self.deploy_bin[self.l, t]) * parameters.min_fallowing_periods for t in self.t_size)
                + 
                gp.quicksum(gp.quicksum(dual_variables[s].rho_2[f, t] * self.y[f, self.l, t] for f in range(self.f_size)) for t in self.t_size)
                +
                gp.quicksum(dual_variables[s].rho_3[t] for t in range(self.t_size))
                +
                dual_variables[s].rho_4
                +
                gp.quicksum(gp.quicksum(dual_variables[s].rho_5[t_hat, t] * self.sites[self.l].MAB_capacity for t in range(self.t_size)) for t_hat in range(self.t_size))
                + 
                gp.quicksum(dual_variables[s].rho_6[t] for t in range(self.t_size))
                + 
                gp.quicksum(dual_variables[s].rho_7[t] for t in range(self.t_size))
            )for s in range(self.s_size)
        )

    def get_variable_values(self):
        y_values = []
        deploy_bin_values = []
        deploy_type_bin_values = []
        for f in range(self.f_size):
            y_values.append([])
            deploy_type_bin_values.append([])
            for t in range(self.t_size):
                y_values[f].append(self.y[self.l, f, t].getAttr("x"))
                deploy_type_bin_values[f].append(self.deploy_type_bin[self.l, f, t].getAttr("x"))
        for t in range(self.t_size):
            deploy_bin_values.append(self.deploy_bin[self.l, t].getAttr("x"))
        return LShapedMasterProblemVariables(self.l, y_values, deploy_bin_values, deploy_type_bin_values)
        
def test():
    problem = LShapedMasterProblem(sites.short_sites_list, 0)
    problem.initialize_model()
    problem.solve()
    values = problem.get_variable_values()
    print(values.y, values.l, values.deploy_bin)

#test()
