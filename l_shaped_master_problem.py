import gurobipy as gp
from gurobipy import GRB
import initialization.parameters as parameters
from data_classes import LShapedMasterProblemVariables, CGDualVariablesFromMaster

class LShapedMasterProblem():
    def __init__(self, 
                 site, 
                 site_index,
                 configs,
                 input_data
                 ):
        self.configs = configs
        cg_dual_variables = CGDualVariablesFromMaster(configs)
        self.input_data = input_data
        self.site = site
        self.l = site_index
        self.s_size = configs.NUM_SCENARIOS
        self.scenario_probabilities = configs.SCENARIO_PROBABILITIES
        self.f_size = configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.growth_sets = self.site.growth_sets
        self.smolt_weights = parameters.smolt_weights
        self.cg_dual_variables = cg_dual_variables


    def initialize_model(self, node_label, cg_dual_variables):
        self.cg_dual_variables = cg_dual_variables
        self.model = gp.Model(f"L-shaped master problem model")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("MIPFocus", 3)
        self.model.setParam("NumericFocus", 3)
        self.model.setParam("IntegralityFocus", 1)
        self.model.setParam("MIPGap", 10**(-5))
        self.declare_variables()
        self.set_objective()
        self.add_initial_condition_constraint()
        self.add_smolt_deployment_constraints()
        self.add_valid_inequality()
        self.add_branching_constraints(node_label)
        self.add_initial_theta_constraint()
    
    def solve(self):
        self.model.optimize()
    
    def declare_variables(self):
        """
        Declares the decision variables for the model, theta, y, and the binary deployment variables.
        :return:
        """
        self.theta = self.model.addVars(self.s_size, vtype=GRB.CONTINUOUS, lb=0, name="theta")
        self.y = self.model.addVars(self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.deploy_bin = self.model.addVars(self.t_size, vtype=GRB.BINARY, name="deploy_bin") # 1 if smolt of any type is deplyed in t NOTE: No l-index as master problem for each l
        self.deploy_type_bin = self.model.addVars(self.f_size, self.t_size, vtype=GRB.BINARY, name="deploy_type_bin") # 1 if smolt type f is deployed in t NOTE: No l-index as master problem for each l

    def set_objective(self):
        """
        Sets the objective of the gurobi model
        :return:
        """
        self.model.setObjective(
            gp.quicksum(
                self.theta[s] for s in range(self.s_size)
            ) - self.cg_dual_variables.v_l[self.l]
            ,GRB.MAXIMIZE
        )
    
    def add_initial_condition_constraint(self): 
        if self.site.init_biomass > 1:
            self.model.addConstr(
                self.y[0,0] == self.site.init_biomass,
                name="Initial Condition"
            )
            self.model.addConstr(
                self.deploy_bin[0] == 1,
                name="Initial Condition bin"
            )

    def add_smolt_deployment_constraints(self):
        """
        Adds the constraints for deployment of smolt. Limiting the number of smolt deployed in any period.

        :return:
        """
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
            gp.quicksum(self.y[f, t] for f in range(self.f_size)) <= parameters.smolt_deployment_upper_bound * self.deploy_bin[t]

            # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
            for t in range(1, self.t_size)
        )
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
            gp.quicksum(self.y[f, t] for f in range(self.f_size)) >= parameters.smolt_deployment_lower_bound * self.deploy_bin[t]
            for t in range(1, self.t_size)
        )
        self.model.addConstrs(  # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
            self.y[f, t] >= parameters.smolt_deployment_lower_bound * self.deploy_type_bin[f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
        )
        self.model.addConstrs(
            # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
            parameters.smolt_deployment_upper_bound * self.deploy_type_bin[f, t] >=  self.y[f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
        )

        if self.site.init_biomass < 0.01:
            #This if statement imposes the biomass limitation in period 0, if there is no initial biomass at the site.
            #If there is biomass at the site in period 0, the deployed biomass will be limited by the initial condition.


            self.model.addConstr(
                # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
                gp.quicksum(self.y[f, 0] for f in range(self.f_size)) <= parameters.smolt_deployment_upper_bound * self.deploy_bin[0]
                , name="Deploy limit if no init biomass at site"
            )
            self.model.addConstr(
                # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
                gp.quicksum(self.y[f, 0] for f in range(self.f_size)) >= parameters.smolt_deployment_lower_bound * self.deploy_bin[0]
                , name="Deploy limit if no init biomass at site"
            )
            self.model.addConstrs(
                # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
                self.y[f, 0] >= parameters.smolt_deployment_lower_bound * self.deploy_type_bin[f, 0]
                for f in range(self.f_size)
            )
            self.model.addConstrs(
                # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
                parameters.smolt_deployment_upper_bound * self.deploy_type_bin[f, 0] >= self.y[f, 0]
                for f in range(self.f_size)
            )

    def add_optimality_cuts(self, dual_variables):
        self.model.addConstrs(
            self.theta[s] <=
            (
                gp.quicksum(dual_variables[s].rho_1[t-parameters.min_fallowing_periods] * (1 - self.deploy_bin[t]) * parameters.min_fallowing_periods for t in range(parameters.min_fallowing_periods, self.t_size))
                + 
                gp.quicksum(gp.quicksum(dual_variables[s].rho_2[f][t] * self.y[f, t] for f in range(self.f_size)) for t in range(self.t_size))
                +
                gp.quicksum(dual_variables[s].rho_3[t] for t in range(self.t_size - parameters.max_fallowing_periods))
                +
                gp.quicksum(dual_variables[s].rho_4[t-parameters.min_fallowing_periods] * (1 - self.deploy_bin[t]) for t in range(parameters.min_fallowing_periods, self.t_size))
                +
                gp.quicksum(gp.quicksum(dual_variables[s].rho_5[t_hat][t] * self.site.MAB_capacity for t in range(min(t_hat + parameters.max_periods_deployed, self.t_size + 1)-t_hat)) for t_hat in range(self.t_size))
                + 
                gp.quicksum(dual_variables[s].rho_6[t] for t in range(self.t_size))
                + 
                gp.quicksum(dual_variables[s].rho_7[t] for t in range(self.t_size))
                +
                gp.quicksum(dual_variables[s].rho_8[t_hat][t] for t in range(self.t_size)for t_hat in range(self.t_size))
            )
            for s in range(self.s_size)
        )

    def add_valid_inequality(self):
        bigM = parameters.valid_ineqaulity_lshaped_master_bigM
        self.model.addConstrs(
            gp.quicksum(
                self.deploy_bin[tau] for tau in range(t + 1, min(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {s}")][t] + parameters.min_fallowing_periods + 2, self.t_size))
            ) <= (1 - self.deploy_bin[t])*bigM #50 should be and adequately large bigM
            for s in range(self.s_size)
            for f in range(self.f_size)
            for t in range(self.t_size)
        )


    def add_initial_theta_constraint(self):
        self.model.addConstr(
            gp.quicksum(
                self.theta[s] for s in range(self.s_size)
            ) <= 1000000000
            , name="Theta init constraint"
        )

    def get_variable_values(self):
        """
        Iterates through all variables in the solution and returns a data class containing their values
        :return:
        """

        #Initializing empty lists, to store the variable values
        y_values = []
        deploy_bin_values = []
        deploy_type_bin_values = []
        #Iterating through the available smolt types
        for f in range(self.f_size):
            #Adding an empty list to the lists
            y_values.append([])
            deploy_type_bin_values.append([])
            #Iterating through through all time periods for the given smolt type
            for t in range(self.t_size):
                #Appending the y and deploy variables to the 2-D list for the given smolt type and period

                y_values[f].append(round(self.y[f, t].getAttr("x"), 1))
                deploy_type_bin_values[f].append(round(self.deploy_type_bin[f, t].getAttr("x"), 1))
        for t in range(self.t_size):
            #Appending the deploy binary values to the list
            deploy_bin_values.append(round(self.deploy_bin[t].getAttr("x"), 1))
        #Returns a data_class with the stores variables
        return LShapedMasterProblemVariables(self.l, y_values, deploy_bin_values, deploy_type_bin_values)


    def add_branching_constraints(self, node_label):
        for index in node_label.up_branching_indices[self.l]:
            self.model.addConstr(
                self.deploy_bin[index] == 1
                , name="branch-up"
            )
       
        for index in node_label.down_branching_indices[self.l]:
            self.model.addConstr(
                self.deploy_bin[index] == 0
                ,name="branch-down"
            )

    def print_variable_values(self):
        variables = self.get_variable_values()
        variables.write_to_file(self.configs)

    def get_deploy_period_list(self):
        deploy_period_list = [t for t in range(0, self.t_size) if self.deploy_bin[t].x == 1]
        return deploy_period_list
        
