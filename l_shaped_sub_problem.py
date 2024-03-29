import pandas as pd
from initialization.input_data import InputData
import initialization.configs
from model import Model
import gurobipy as gp
from gurobipy import GRB
import initialization.sites as sites
from data_classes import LShapedMasterProblemVariables

class LShapedSubProblem(Model):
    def __init__(self,
                 scenario: int,                                 #An int between 0 and s where s is the number of scenarios. Denotes the scenario in this subporblem
                 location: int,                                 #An int between 0 and L, where l is the number of locations. Denotes the location of this subproblem
                 fixed_variables: LShapedMasterProblemVariables,
                 site_objects,
                 MAB_shadow_prices_df=pd.DataFrame(),
                 EOH_shadow_prices_df=pd.DataFrame(), 
                 input_data=InputData(), 
                 parameters=initialization.parameters, 
                 scenario_probabilities=initialization.configs.SCENARIO_PROBABILITIES):
        
        self.fixed_variables = fixed_variables
        self.scenario = scenario
        self.location = location
        super().__init__(site_objects, MAB_shadow_prices_df, EOH_shadow_prices_df, input_data, parameters, scenario_probabilities)

    
    def solve(self):
        #TODO: Implement
        self.model = gp.Model("LShapedSubProblem")
        self.declare_variables()

        #1. Setobjective
        self.add_objective()
        self.add_fallowing_constraints()
        self.add_biomass_development_constraints()
        self.add_w_forcing_constraint()
        self.add_MAB_requirement_constraint()
        self.add_inactivity_constraint()
        self.add_harvest_forcing_constraints()
        self.add_employment_bin_forcing_constraints()

        #2. Add constraints

        #3. Optimize
        self.model.optimize()

        for v in self.model.getVars():
            print(f'{v.varName} {v.X}')

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
        self.z_slack_2 = self.model.addVars(self.t_size, self.t_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="z_slack_2")
        self.z_slack_3 = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="z_slack_3")
        # Declaring, the binary variables from the original problem as continuous due to the LP Relaxation
        # These must be continous for us to be able to fetch the dual values out
        self.harvest_bin = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, name = "harvest_bin", lb=0, ub=1)
        self.employ_bin = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, name = "Employ bin", lb =0, ub=1)
        self.employ_bin_granular = self.model.addVars(self.t_size, self.t_size, vtype=GRB.CONTINUOUS, name="Employ bin gran", lb=0, ub=1)
        #TODO: Implement with only continous variables


    """
    Objective
    """
    def add_objective(self):
        Penalty_parameter = 100000000000
        self.model.setObjective(
            gp.quicksum(self.w[f,t_hat,t]
                        for f in range(self.f_size)
                        for t_hat in range(self.t_size)
                        for t in range(self.growth_sets[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat],
                                       min(t_hat + self.parameters.max_periods_deployed, self.t_size))
                        )

            - Penalty_parameter * gp.quicksum(self.z_slack_1[t] for t in range(self.t_size)) #TODO: Change to the actual set used
            - Penalty_parameter * gp.quicksum(self.z_slack_2[t_hat, t] for t_hat in range(self.t_size) for t in range(t_hat, self.t_size))
            - Penalty_parameter * gp.quicksum(self.z_slack_3[t] for t in range(self.t_size))
            # NOTE: This is not the range specified in the formulation, but it should work since
            # the slack variable will always be 0 if it can with this formulation of the max problem.
            #TODO: Change to a more specific range if necesarry.
            , GRB.MAXIMIZE
        )

    """
    Constraints
    """

    #74
    def add_fallowing_constraints(self): #Fallowing constraint
        # Fixed
        self.model.addConstrs(
            gp.quicksum(
                self.employ_bin[tau] for tau in range(t - self.parameters.min_fallowing_periods, t)
            )
            - self.z_slack_1[t]
            <= self.parameters.min_fallowing_periods * (1 - self.fixed_variables.deploy_bin[t])
            for t in range(self.parameters.min_fallowing_periods, self.t_size)
        )

        #TODO: Figure out if this constraint needs to be considered when passing the sensitivities up!
        self.model.addConstr(
            # This is an additional constraint - ensuring that only 1 deployment happens during the initial possible deployment period TODO: See if this needs to be implemented in the math model
            gp.quicksum(self.fixed_variables.deploy_bin[t] for t in range(self.parameters.min_fallowing_periods)) <= 1
            , name="initial constraint"
        )

        #TODO: Implement with slack variable and fixed gamma (74)
        pass

    #75
    def add_inactivity_constraint(self):
        self.model.addConstrs(
            # This is the constraint (5.7) - ensuring that the site is not inactive longer than the max fallowing limit
            gp.quicksum(self.employ_bin[tau] for tau in
                        range(t, min(t + self.parameters.max_fallowing_periods, self.t_size))) >= 1
            # The sum function and therefore the t set is not implemented exactly like in the mathematical model, but functionality is the same
            for t in range(self.t_size)
        )

    #76 - 77
    def add_harvest_forcing_constraints(self):
        # Fixed
        self.model.addConstrs(
            # This is the first part of constraint (5.8) - which limits harvest in a single period to an upper limit
            gp.quicksum(
                self.w[f, t_hat, t] for f in range(self.f_size) for t_hat in
                        range(t)) - self.parameters.max_harvest * self.harvest_bin[t] <= 0
            for t in range(self.t_size)
        )

        # Fixed
        self.model.addConstrs(
            # This is the second part of constraint (5.8) - which limits harvest in a single period to a lower limit
                self.parameters.min_harvest * self.harvest_bin[t] - gp.quicksum(self.w[f, t_hat, t] for f in range(self.f_size) for t_hat in range(t)) <= 0
            for t in range(self.t_size)
        )
        pass

    #78 - 81
    def add_biomass_development_constraints(self):
        self.model.addConstrs(  # This is constraint (5.9) - which ensures that biomass x = biomass deployed y
            self.x[f, t, t] == self.fixed_variables.y[f][t]
            for f in range(self.f_size)
            for t in range(self.t_size)
        )

        # Fixed
        self.model.addConstrs(
            # This represents the constraint (5.10) - which ensures biomass growth in the growth period
            self.x[f, t_hat, t + 1] == (1 - self.parameters.expected_production_loss) * self.x[
                f, t_hat, t] *
            self.growth_factors[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}", t_hat)][t]
            for t_hat in range(self.t_size - 1)
            for f in range(self.f_size)
            for t in
            range(min(t_hat, self.t_size),
                  min(self.growth_sets[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size))
        )

        # Fixed
        self.model.addConstrs(
            # This is the constraint (5.11) - Which tracks the biomass employed in the harvest period
            self.x[f, t_hat, t + 1] == (1 - self.parameters.expected_production_loss) * self.x[
                f, t_hat, t] *
            self.growth_factors[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}", t_hat)][t] - self.w[f, t_hat, t]
            for t_hat in range(self.t_size)
            for f in range(self.f_size)
            for t in range(min(self.growth_sets[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size),
                           min(t_hat + self.parameters.max_periods_deployed, self.t_size))

        )

    def add_employment_bin_forcing_constraints(self):
        self.model.addConstrs(
            self.employ_bin_granular[t_hat, t] - gp.quicksum(
                self.x[f, t_hat, t] for f in range(self.f_size)) <= 0
            for t_hat in range(self.t_size)
            for t in range(self.t_size)
        )

        self.model.addConstrs(
            gp.quicksum(self.x[f, t_hat, t] for f in range(self.f_size)) - self.employ_bin_granular[
                t_hat, t] * self.parameters.bigM <= 0
            for t_hat in range(self.t_size)
            for t in range(self.t_size)
        )

        self.model.addConstrs(
            self.employ_bin[t] - gp.quicksum(
                self.employ_bin_granular[t_hat, t] for t_hat in range(self.t_size)) == 0
            for t in range(self.t_size)
        )

    #84

    def add_MAB_requirement_constraint(self):
        self.model.addConstrs(
            gp.quicksum(self.x[ f, t_hat, t] for f in range(self.f_size)) - self.z_slack_2[t_hat,t] <= self.sites[self.location].MAB_capacity
            for t_hat in range(self.t_size)
            for t in range(t_hat, min(t_hat + self.parameters.max_periods_deployed, self.t_size + 1))
        )
        #TODO: Implement with slack variable (82)
        pass


    def add_w_forcing_constraint(self):
        self.model.addConstrs(
            # TODO:This is a forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[f, t_hat, t] == 0
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in
            range(0, min(self.growth_sets[self.location].loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size))
        )

        self.model.addConstrs(
            # TODO:This is a second forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[f, t_hat, t] == 0
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(min(t_hat + self.parameters.max_periods_deployed, self.t_size), self.t_size)
        )

    def add_x_forcing_constraint(self):  # TODO: check if used or remove
        self.model.addConstrs(
            self.x[f, t_hat, t] <= 0
            for t_hat in range(self.t_size)
            for t in range(0, t_hat)
            for f in range(self.f_size)
        )

    def get_dual_values(self):
        #TODO: Implement
        pass

if __name__ == "__main__":
    y = [[0.0 for i in range(60)]]
    y[0][0] = 0
    y[0][30] = 1000 *100
    y[0][8] = 1000 * 100

    fixed_variables = LShapedMasterProblemVariables(
        l=1,
        y = y,
        deploy_bin = [0 for i in range(60)],
        deploy_type_bin= [[0 for i in range(60)]]
    )
    model = LShapedSubProblem(location=1, scenario=0, site_objects=initialization.sites.SITE_LIST, fixed_variables=fixed_variables)
    model.solve()

