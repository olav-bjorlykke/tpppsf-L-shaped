import pandas as pd
from initialization.input_data import InputData
import initialization.configs
from model import Model
import gurobipy as gp
from gurobipy import GRB
import initialization.sites as sites

class LShapedSubProblem(Model):
    def __init__(self,
                 scenario: int,                                 #An int between 0 and s where s is the number of scenarios. Denotes the scenario in this subporblem
                 location: int,                                 #An int between 0 and L, where l is the number of locations. Denotes the location of this subproblem
                 site_objects,
                 MAB_shadow_prices_df=pd.DataFrame(),
                 EOH_shadow_prices_df=pd.DataFrame(), 
                 input_data=InputData(), 
                 parameters=initialization.parameters, 
                 scenario_probabilities=initialization.configs.SCENARIO_PROBABILITIES):
        
        self.scenario = scenario
        self.location = location
        super().__init__(site_objects, MAB_shadow_prices_df, EOH_shadow_prices_df, input_data, parameters, scenario_probabilities)

    
    def solve(self):
        #TODO: Implement
        self.model = gp.Model("LShapedSubProblem")
        self.declare_variables()

        #1. Setobjective
        self.add_objective()

        #2. Add constraints

        #3. Optimize
        self.model.optimize()

        pass
    
    def declare_variables(self):
        """
        Declares variables to be used in the model.

        :return:
        """
        #The biomass tracking variable
        #s and l removed as indices due to them being fixed in every sub problem
        self.x = self.model.addVars(self.f_size, self.t_size, self.t_size + 1,
                                    vtype=GRB.CONTINUOUS, lb=0, name="X")
        # The harvest variable
        # s and l removed as indices due to them being fixed in every sub problem
        self.w = self.model.addVars(self.f_size, self.t_size, self.t_size,
                                    vtype=GRB.CONTINUOUS, lb=0, name="W")
        #Declaring slack variables
        self.z_slack_1 = self.model.addVars(self.t_size,vtype=GRB.CONTINUOUS, lb = 0, name = "z_slack_1")
        self.z_slack_2 = self.model.addVars(self.t_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="z_slack_2")
        # Declaring, the binary variables from the original problem as continuous due to the LP Relaxation
        # These must be continous for us to be able to fetch the dual values out
        self.harvest_bin = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS)
        self.employ_bin = self.model.addVars(self.t_size, self.t_size, vtype=GRB.CONTINUOUS)
        #TODO: Implement with only continous variables
        pass

    def set_fixed_variables(self):

    def add_objective(self):
        Penalty_parameter = 10000000000
        self.model.setObjective(
            gp.quicksum(self.w[f,t_hat,t]
                        for f in range(self.f_size)
                        for t_hat in range(self.t_size)
                        for t in range(self.growth_sets[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat],
                                       min(t_hat + self.parameters.max_periods_deployed, self.t_size))
                        )

            - Penalty_parameter * gp.quicksum(self.z_slack_1[t] for t in range(self.t_size)) #TODO: Change to the actual set used

            - Penalty_parameter * gp.quicksum(self.z_slack_2[t_hat, t]
                                              for t_hat in range(self.t_size)
                                              for t in range(t_hat, self.t_size))
            # NOTE: This is not the range specified in the formulation, but it should work since
            # the slack variable will always be 0 if it can with this formulation of the max problem.
            #TODO: Change to a more specific range if necesarry.

        )

    def add_fallowing_constraints(self): #Fallowing constraint
        # Fixed
        self.model.addConstrs(
            gp.quicksum(
                self.employ_bin[tau] for tau in range(t - self.parameters.min_fallowing_periods, t)
            )
            - self.z_slack_1[t]
            <= self.parameters.min_fallowing_periods * (1 - self.deploy_bin[t]) #TODO: Replace with fixed gamma
            for t in range(self.parameters.min_fallowing_periods, self.t_size)
        )

        #TODO: Figure out if this constraint needs to be considered when passing the sensitivities up!
        self.model.addConstrs(
            # This is an additional constraint - ensuring that only 1 deployment happens during the initial possible deployment period TODO: See if this needs to be implemented in the math model
            gp.quicksum(self.deploy_bin[t] for t in range(self.parameters.min_fallowing_periods)) <= 1
        )

        #TODO: Implement with slack variable and fixed gamma (74)
        pass

    def add_biomass_development_constraints(self):
        #TODO: Implement with fixed y (77)
        pass

    def add_MAB_requirement_constraint(self):
        #TODO: Implement with slack variable (82)
        pass

    def get_dual_values(self):
        #TODO: Implement
        pass

if __name__ == "__main__":
    model = LShapedSubProblem(location=0, scenario=0, site_objects=initialization.sites.SITE_LIST)
    model.solve()