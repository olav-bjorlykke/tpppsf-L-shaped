from initialization.input_data import InputData
import initialization.parameters as parameters
from initialization.sites import Site
from model import Model
import gurobipy as gp
from gurobipy import GRB
from data_classes import LShapedMasterProblemVariables, LShapedSubProblemDualVariables, CGDualVariablesFromMaster
import pandas as pd

class LShapedSubProblem(Model):
    def __init__(self,
                 scenario: int,
                 site: Site,
                 site_index: int,                                 
                 fixed_variables: LShapedMasterProblemVariables,
                 cg_dual_variables: CGDualVariablesFromMaster,
                 configs,
                 ):
        self.site = site
        self.scenario = scenario
        self.location = site_index
        self.fixed_variables = fixed_variables
        self.cg_dual_variables = cg_dual_variables
        self.input_data = InputData(configs)
        self.configs = configs
        self.s_size = configs.NUM_SCENARIOS
        self.f_size = configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.growth_factors = self.site.growth_per_scenario_df
        self.growth_sets = self.site.growth_sets
        self.smolt_weights = parameters.smolt_weights


    """
    Model declaration and initialization functions
    """
    def initialize_model(self):
        self.model = gp.Model("LShapedSubProblem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('DualReductions', 0)
        self.declare_variables()
        #1. Setobjective
        self.add_cg_dual_objective()
        #2. Add constraints
        self.add_fallowing_constraints()
        self.add_biomass_development_constraints()
        self.add_w_forcing_constraint()
        self.add_MAB_requirement_constraint()
        self.add_UB_constraints()
        self.add_inactivity_constraint()
        self.add_harvest_constraints()
        self.add_employment_bin_forcing_constraints()
        self.add_valid_inequality_sub_problem()
        #self.add_x_forcing_constraint()

    def update_model(self, fixed_variables):
        self.model.setParam("MIPFocus", 0)
        self.model.setParam("NumericFocus", 0)
        self.fixed_variables = fixed_variables
        self.model.remove(self.model.getConstrs())
        self.add_fallowing_constraints()
        self.add_biomass_development_constraints()
        self.add_w_forcing_constraint()
        #self.add_x_forcing_constraint()
        self.add_MAB_requirement_constraint_lp()
        self.add_UB_constraints()
        self.add_inactivity_constraint()
        self.add_harvest_constraints()
        self.add_employment_bin_forcing_constraints()
        self.add_valid_inequality_sub_problem()

    def update_model_to_mip(self, fixed_variables):
        self.model.setParam("MIPFocus", 3)
        self.model.setParam("NumericFocus", 3)
        self.model.remove(self.model.getConstrs())
        self.model.remove(self.model.getVars())
        self.declare_mip_variables()
        self.add_mip_objective()
        self.fixed_variables = fixed_variables
        self.add_fallowing_constraints()
        self.add_biomass_development_constraints()
        self.add_w_forcing_constraint()
        #self.add_x_forcing_constraint()
        self.add_MAB_requirement_constraint()
        self.add_UB_constraints()
        self.add_inactivity_constraint()
        self.add_harvest_constraints()
        self.add_employment_bin_forcing_constraints()
        self.add_valid_inequality_sub_problem()

    def solve(self):
        self.model.optimize()
    def declare_variables(self):
        """
        Declares variables to be used in the model.
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
        self.z_slack_3 = self.model.addVars(self.t_size,vtype=GRB.CONTINUOUS, lb = 0, name = "z_slack_3" )
        # Declaring, the binary variables from the original problem as continuous due to the LP Relaxation
        # These must be continous for us to be able to fetch the dual values out
        self.harvest_bin = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, name = "harvest_bin", lb=0) # UB moved to constraints to get dual variable value
        self.employ_bin = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, name = "Employ bin", lb =0) # UB moved to constraints to get dual variable value
        self.employ_bin_granular = self.model.addVars(self.t_size, self.t_size, vtype=GRB.CONTINUOUS, name="Employ bin gran", lb=0, ub=1)
    def declare_mip_variables(self):
        """
                Declares variables to be used in the model.
                """
        # The biomass tracking variable
        # s and l removed as indices due to them being fixed in every sub problem
        self.x = self.model.addVars(self.f_size, self.t_size, self.t_size + 1, vtype=GRB.CONTINUOUS, lb=0, name="X")
        # The harvest variable
        # s and l removed as indices due to them being fixed in every sub problem
        self.w = self.model.addVars(self.f_size, self.t_size, self.t_size,vtype=GRB.CONTINUOUS, lb=0, name="W")
        # Declaring slack variables
        self.z_slack_1 = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=0, name="z_slack_1")
        self.z_slack_2 = self.model.addVars(self.t_size, self.t_size + 1, vtype=GRB.CONTINUOUS, lb=0, ub=0, name="z_slack_2")
        self.z_slack_3 = self.model.addVars(self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=0, name="z_slack_3")
        # Declaring, the binary variables from the original problem as continuous due to the LP Relaxation
        # These must be continous for us to be able to fetch the dual values out
        self.harvest_bin = self.model.addVars(self.t_size, vtype=GRB.BINARY, name="harvest_bin", lb=0)
        self.employ_bin = self.model.addVars(self.t_size, vtype=GRB.BINARY, name="Employ bin", lb=0)
        self.employ_bin_granular = self.model.addVars(self.t_size, self.t_size, vtype=GRB.BINARY, name="Employ bin gran", lb=0)


    """
    Objective
    """
    def add_mip_objective(self):
        self.model.setObjective(
            gp.quicksum(
                self.configs.SCENARIO_PROBABILITIES[self.scenario] *
                gp.quicksum(self.w[f, t_hat, t]
                            for t in range(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat],
                                  min(t_hat + parameters.max_periods_deployed, self.t_size))
                )
                - gp.quicksum(
                    self.x[f, t_hat, t] * self.cg_dual_variables.u_MAB[t][self.scenario]
                    for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, self.t_size + 1))
                )
                - self.x[f, t_hat, parameters.number_periods] * self.cg_dual_variables.u_EOH[self.scenario]
                for f in range(self.f_size)
                for t_hat in range(self.t_size)
            ),GRB.MAXIMIZE
        )

    def add_cg_dual_objective(self):
        penalty_parameter = parameters.penalty_parameter_L_sub #This should not be very high -> It will lead to numeric instability
        self.model.setObjective(
            gp.quicksum(
                self.configs.SCENARIO_PROBABILITIES[self.scenario] *
                gp.quicksum(
                    self.w[f, t_hat, t] for t in range(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat],
                                    min(t_hat + parameters.max_periods_deployed, self.t_size))
                )
                - gp.quicksum(
                    self.x[f, t_hat, t] * self.cg_dual_variables.u_MAB[t][self.scenario]
                    for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, self.t_size + 1))
                )
                - self.x[f, t_hat, parameters.number_periods] * self.cg_dual_variables.u_EOH[self.scenario]
                for f in range(self.f_size)
                for t_hat in range(self.t_size)
            )
            - penalty_parameter * gp.quicksum(self.z_slack_1[t] for t in range(self.t_size))
            - penalty_parameter * 5000 * gp.quicksum(self.z_slack_2[t_hat, t] for t_hat in range(self.t_size) for t in range(t_hat, self.t_size))
            - penalty_parameter * gp.quicksum(self.z_slack_3[t] for t in range(self.t_size))
            , GRB.MAXIMIZE
        )
    """
    Constraints
    """
    def add_fallowing_constraints(self): #Fallowing constraint
        # Fixed
        self.model.addConstrs((
            gp.quicksum(
                self.employ_bin[tau] for tau in range(t - parameters.min_fallowing_periods, t)
            )
            - self.z_slack_1[t]
            <= parameters.min_fallowing_periods * (1 - self.fixed_variables.deploy_bin[t])
            for t in range(parameters.min_fallowing_periods, self.t_size)), name="fallowing_constriants_1"
        )
    def add_inactivity_constraint(self):
        self.model.addConstrs((
            # This is the constraint (5.7) - ensuring that the site is not inactive longer than the max fallowing limit
            gp.quicksum(self.employ_bin[tau] for tau in
                        range(t, min(t + parameters.max_fallowing_periods, self.t_size))) + self.z_slack_3[t] >= 1
            # The sum function and therefore the t set is not implemented exactly like in the mathematical model, but functionality is the same
            for t in range(self.t_size - parameters.max_fallowing_periods)), name="inactivity_constraints"
        )
    def add_harvest_constraints(self):
        self.model.addConstrs(
            gp.quicksum(
                self.w[f, t_hat, t] for f in range(self.f_size) for t_hat in
                        range(t)) - parameters.max_harvest * self.harvest_bin[t] <= 0
            for t in range(self.t_size)
        )
        self.model.addConstrs(
                parameters.min_harvest * self.harvest_bin[t] - gp.quicksum(self.w[f, t_hat, t] for f in range(self.f_size) for t_hat in range(t)) <= 0
            for t in range(self.t_size)
        )
    def add_biomass_development_constraints(self):
        self.model.addConstrs((
            #This tracks the biomass in the deploy period
            self.x[f, t, t] == self.fixed_variables.y[f][t]
            for f in range(self.f_size)
            for t in range(self.t_size)), name="biomass_development_constraints_1"
        )

        self.model.addConstrs(
            #This tracks the biomass in the growth period
            self.x[f, t_hat, t + 1] == (1 - parameters.expected_production_loss) * self.x[
                f, t_hat, t] *
            self.growth_factors.loc[(self.smolt_weights[f], f"Scenario {self.scenario}", t_hat)][t]
            for t_hat in range(self.t_size)
            for f in range(self.f_size)
            for t in
            range(t_hat,
                  min(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size))
        )

        self.model.addConstrs(
            # This tracks the biomass employed in the harvest period
            self.x[f, t_hat, t + 1] == (1 - parameters.expected_production_loss) * self.x[
                f, t_hat, t] *
            self.growth_factors.loc[(self.smolt_weights[f], f"Scenario {self.scenario}", t_hat)][t] - self.w[f, t_hat, t]
            for t_hat in range(self.t_size)
            for f in range(self.f_size)
            for t in range(min(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size),
                            self.t_size)

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
                t_hat, t] * parameters.bigM <= 0
            for t_hat in range(self.t_size)
            for t in range(self.t_size)
        )

        self.model.addConstrs(
            self.employ_bin[t] - gp.quicksum(
                self.employ_bin_granular[t_hat, t] for t_hat in range(self.t_size)) == 0
            for t in range(self.t_size)
        )

        self.model.addConstrs(
            self.employ_bin_granular[t_hat, t] == 0
            for t_hat in range(self.t_size)
            for t in range(t_hat)
        )
        
    def add_valid_inequality_sub_problem(self):
        self.model.addConstrs((
            0.5 * (self.employ_bin[t - 1] + self.employ_bin[t - 2]) + self.fixed_variables.deploy_bin[t] <= 1
            for t in range(parameters.min_fallowing_periods, self.t_size)
        ), name="valid_inequality")

    def add_MAB_requirement_constraint(self):
        self.model.addConstrs((
            gp.quicksum(self.x[f, t_hat, t] for f in range(self.f_size)) - self.z_slack_2[t_hat,t] <= self.site.MAB_capacity
            for t_hat in range(self.t_size)
            for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, self.t_size +1))
        ), name="MAB_constraints_mip"
        )
    def add_MAB_requirement_constraint_lp(self):
        self.model.addConstrs((
            gp.quicksum(self.x[f, t_hat, t] for f in range(self.f_size)) - self.z_slack_2[t_hat,t] <= self.site.MAB_capacity * 0.995
            for t_hat in range(self.t_size)
            for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, self.t_size +1))
        ), name="MAB_constraints"
        )
    def add_UB_constraints(self):
        self.model.addConstrs((
            self.harvest_bin[t] <= 1 for t in range(self.t_size)), name="harvest_bin_UB"
        )
        self.model.addConstrs((
            self.employ_bin[t] <= 1 for t in range(self.t_size)), name="employ_bin_UB"
        )
        self.model.addConstrs((
            self.employ_bin_granular[t_hat, t] <= 1
            for t in range(self.t_size)
            for t_hat in range(self.t_size)
        ), name="employ_bin_gran_UB"
        )


    def add_w_forcing_constraint(self):
        self.model.addConstrs(
            # TODO:This is a forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[f, t_hat, t] == 0
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in
            range(0, min(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {self.scenario}")][t_hat], self.t_size))
        )

        self.model.addConstrs(
            # TODO:This is a second forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[f, t_hat, t] == 0
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(min(t_hat + parameters.max_periods_deployed, self.t_size), self.t_size)
        )

    def add_x_forcing_constraint(self):#TODO: check if used or remove
        self.model.addConstrs(
            self.x[f, t_hat, t] <= 0
            for t_hat in range(self.t_size)
            for t in range(0, t_hat)
            for f in range(self.f_size)
        )

        self.model.addConstrs(
            self.x[f, t_hat, t] <= 0
            for t_hat in range(self.t_size)
            for t in range(min(t_hat + parameters.max_periods_deployed, self.t_size + 1), self.t_size + 1)
            for f in range(self.f_size)
        )



    """
    Print and export values constraints
    """
    def get_dual_values(self):
        rho_1 = []
        rho_2 = []
        rho_3 = []
        rho_4 = [] 
        rho_5 = []
        rho_6 = []
        rho_7 = []
        rho_8 = []
        for t in range(parameters.min_fallowing_periods, self.t_size):
            rho_1.append(round(self.model.getConstrByName(f"fallowing_constriants_1[{t}]").getAttr("Pi"), 5))
            rho_4.append(round(self.model.getConstrByName(f"valid_inequality[{t}]").getAttr("Pi"), 5))
        for f in range(self.f_size):
            rho_2.append([])
            for t in range(self.t_size):
                rho_2[f].append(round(self.model.getConstrByName(f"biomass_development_constraints_1[{f},{t}]").getAttr("Pi"), 5))
        for t in range(self.t_size - parameters.max_fallowing_periods):
            rho_3.append(round(self.model.getConstrByName(f"inactivity_constraints[{t}]").getAttr("Pi"), 5))
        for t in range(self.t_size):
            rho_6.append(round(self.model.getConstrByName(f"harvest_bin_UB[{t}]").getAttr("Pi"), 5))
            rho_7.append(round(self.model.getConstrByName(f"employ_bin_UB[{t}]").getAttr("Pi"), 5))
        for t_hat in range(self.t_size):
            rho_5.append([])
            for t in range(t_hat, min(t_hat + parameters.max_periods_deployed, self.t_size + 1)):
                rho_5[t_hat].append(round(self.model.getConstrByName(f"MAB_constraints[{t_hat},{t}]").getAttr("Pi"), 5))
            rho_8.append([])
            for t in range(self.t_size):
                rho_8[t_hat].append(round(self.model.getConstrByName(f"employ_bin_gran_UB[{t_hat},{t}]").getAttr("Pi"), 5))
        return LShapedSubProblemDualVariables(rho_1, rho_2, rho_3, rho_4, rho_5, rho_6, rho_7, rho_8)

    def print_variable_values(self):
        f_list = []
        for f in range(self.f_size):
            t_hat_list = []
            for t_hat in range(self.t_size):
                t_list = []
                for t in range(self.t_size):
                    x = self.x[f,t_hat, t].x
                    w = self.w[f,t_hat, t].x
                    employ_bin = self.employ_bin[t].x
                    employ_bin_gran = self.employ_bin_granular[t_hat,t].x
                    harvest_bin = self.harvest_bin[t].x
                    z_slack_1 = self.z_slack_1[t].x
                    z_slack_2 = self.z_slack_2[t_hat,t].x

                    variable_list = [x, w, employ_bin, employ_bin_gran, harvest_bin, z_slack_1, z_slack_2]

                    t_list.append(variable_list)
                colums = ["X", "W", "Employ_bin", "Employ_bin_gran", "Harvest_bin", "z_slack_1", "z_slack_2"]
                t_hat_list.append(pd.DataFrame(t_list, columns=colums, index=[i for i in range(self.t_size)]))
            f_list.append(pd.concat(t_hat_list, keys=[i for i in range(self.t_size)]))
        df = pd.concat(f_list, keys=[i for i in range(self.f_size)])
        df_filtered = df.loc[~(df[["X", "W", "Employ_bin_gran"]] == 0).all(axis=1)]
        df_filtered.to_excel(f"{self.configs.OUTPUT_DIR}variable_values{self.scenario}.xlsx")
