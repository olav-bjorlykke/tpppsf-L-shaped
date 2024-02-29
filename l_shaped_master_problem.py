import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from initialization.input_data import InputData
import initialization.parameters as parameters
import initialization.configs as configs
from model import Model

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
        self.y = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.BINARY, name="gamma") # 1 if smolt of any type is deplyed in t NOTE: No l-index as master problem for each l
        self.deploy_type_bin = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.BINARY, name="delta") # 1 if smolt type f is deployed in t NOTE: No l-index as master problem for each l 

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(
                self.scenario_probabilities[s] *
                gp.quicksum(
                    self.theta[s] for s in range(self.s_size)
                )
                for s in range(self.s_size)
            )
            ,GRB.MAXIMISE
        )

    def add_optimality_cuts(self, dual_variables): # TODO: Implement with actual rho values based on datastructure generated from sub-problem
        self.model.addConstrs(
            self.theta[s] <= (
                gp.quicksum(dual_variables.rho_1[s, t] * (1 - self.deploy_bin[self.l, t]) * parameters.min_fallowing_periods for t in self.t_size)
                + 
                gp.quicksum(gp.quicksum(dual_variables.rho_2[s, f, t] * self.y[f, self.l, t] for f in range(self.f_size)) for t in self.t_size)
                +
                gp.quicksum(dual_variables.rho_3[s, t] for t in range(self.t_size))
                +
                dual_variables.rho_4[s]
                +
                gp.quicksum(gp.quicksum(dual_variables.rho_5[s, t_hat, t] * self.sites[self.l].MAB_capacity for t in range(self.t_size)) for t_hat in range(self.t_size))
                + 
                gp.quicksum(dual_variables.rho_6[s, t] for t in range(self.t_size))
                + 
                gp.quicksum(dual_variables.rho_7[s, t] for t in range(self.t_size))
            )for s in range(self.s_size)
        )
        
    def get_y_values(self):
        return self.get_deploy_amounts_df()
    
    def get_gamma_values(self):
        return self.get_deploy_period_list()
    
    def get_delta_values(self):
        return self.get_deploy_period_list_per_cohort()
