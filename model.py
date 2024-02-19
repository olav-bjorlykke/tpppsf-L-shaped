import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import numpy as np
from parameters import GlobalParameters
from input_data import InputData
from site_class import Site
from scenarios import Scenarios
import matplotlib.pyplot as plt
import configs
import time

class Model:
    iterations = 0
    solution_index_names = ["Scenario", "Smolt type", "Deploy period", "Period"]
    column_index_names = ["Location", "Scenario", "Smolt type", "Deploy period", "Period"]
    time_in_subproblem = 0


    def __init__(self,
                 site_objects,
                 MAB_shadow_prices_df = pd.DataFrame(),
                 EOH_shadow_prices_df = pd.DataFrame()
                 ):
        #Imported classes, containing parameters and data
        self.input_data = InputData()
        self.parameters = GlobalParameters()
        self.scenario = Scenarios(self.input_data.temperatures_df)

        if isinstance(site_objects,list):
            self.sites = site_objects
        else:
            self.sites = [site_objects]


        #Setting variables to contain the size of sets
        self.f_size = 1  #TODO: declare using the smolt set
        self.t_size = self.parameters.number_periods
        self.s_size = configs.NUM_SCENARIOS  #TODO: len(parameters.scenario_probabilities)
        self.l_size = len(self.sites)

        #Defining some variables from the data objects for easier reference
        self.growth_factors = [site.growth_per_scenario_df for site in self.sites]
        self.smolt_weights = self.parameters.smolt_weights
        self.growth_sets = [site.growth_sets for site in self.sites]
        self.MAB_shadow_prices_df = MAB_shadow_prices_df
        self.EOH_shadow_prices_df = EOH_shadow_prices_df

        #Defining some instance attributes:
        self.iterations = 0
        self.branching_variable_indices_up = []
        self.branching_variable_indices_down = []


    def solve_and_print_model(self):
        self.model = gp.Model(f"Single site solution")

        #Declaing variables
        self.declare_variables()

        #Setting objective
        self.set_objective()

        #Adding constraints
        self.add_smolt_deployment_constraints()
        self.add_fallowing_constraints()
        self.add_inactivity_constraints()
        self.add_harvesting_constraints()
        self.add_biomass_development_constraints()
        self.add_MAB_requirement_constraint()
        self.add_initial_condition_constraint()
        self.add_forcing_constraints()
        self.add_MAB_company_requirement_constraint()
        self.add_end_of_horizon_constraint()
        #self.add_x_forcing_constraint()
        self.add_up_branching_constraints()
        self.add_down_branching_constraints()

        #Running gurobi to optimize model
        self.model.optimize()

        #Printing solution
        if self.model.status == GRB.OPTIMAL:
            self.print_solution_to_excel()
            self.plot_solutions_x_values_per_site()
            self.plot_solutions_x_values_aggregated()
            self.iterations += 1


        #Putting solution into variables for export

    def solve_as_sub_problem(self):
        start_time = time.perf_counter()
        self.model = gp.Model(f"Single site solution")

        # Declaing variables
        self.declare_variables()

        # Setting objective
        self.set_decomped_objective()

        # Adding constraints
        self.add_smolt_deployment_constraints()
        self.add_fallowing_constraints()
        self.add_inactivity_constraints()
        self.add_harvesting_constraints()
        self.add_biomass_development_constraints()
        self.add_MAB_requirement_constraint()
        self.add_initial_condition_constraint()
        self.add_forcing_constraints()
        #TODO: Explain why these are commented out

        # self.add_MAB_company_requirement_constraint()
        # self.add_end_of_horizon_constraint()
        # self.add_x_forcing_constraint()
        self.add_up_branching_constraints()
        self.add_down_branching_constraints()

        # Running gurobi to optimize model
        self.model.optimize()

        if self.model.status != GRB.INFEASIBLE:
            self.plot_solutions_x_values_per_site()

        end_time = time.perf_counter()
        time_to_run_master = end_time - start_time
        print(f"#### time to run: {time_to_run_master}")
        print(f"{self.time_in_subproblem}")
        self.time_in_subproblem += time_to_run_master
        print(f"{self.time_in_subproblem}")


    def create_initial_columns(self):
        self.model = gp.Model(f"Find feasible solution")

        #Telling the model to focus on finding a feasible solution
        self.model.setParam("MIPFocus", 1)
        #Stopping the model after one feasible solution is found
        self.model.setParam('SolutionLimit', 10)

        # Declaing variables
        self.declare_variables()

        # Setting objective
        self.set_objective()

        # Adding constraints
        self.add_smolt_deployment_constraints()
        self.add_fallowing_constraints()
        self.add_inactivity_constraints()
        self.add_harvesting_constraints()
        self.add_biomass_development_constraints()
        self.add_MAB_requirement_constraint()
        self.add_initial_condition_constraint()
        self.add_forcing_constraints()
        self.add_MAB_company_requirement_constraint()
        self.add_end_of_horizon_constraint()
        # self.add_x_forcing_constraint()
        # self.add_up_branching_constraints()
        # self.add_down_branching_constraints()

        # Running gurobi to optimize model
        self.model.optimize()

        # Printing solution
        if self.model.status != GRB.INFEASIBLE:
            #self.plot_solutions_x_values_per_site()
            #self.plot_solutions_x_values_aggregated()
            self.iterations += 1
            return self.get_solution_as_df()
        else:
            return None



    def declare_variables(self):
        self.x = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size +1, self.s_size, vtype=GRB.CONTINUOUS, lb=0, name="X")
        self.y = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, name="y")
        self.w = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, name="W")

        # Declaring the binary decision variables
        self.deploy_type_bin = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.BINARY)
        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.BINARY)
        self.harvest_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.BINARY)
        self.employ_bin = self.model.addVars(self.l_size, self.t_size, self.s_size, vtype=GRB.BINARY)

    def set_objective(self):
        #FIXED
        self.model.setObjective(  # This is the objective (5.2) - which represents the objective for biomass maximization
            gp.quicksum(
                self.scenario.scenario_probabilities[s] *
                gp.quicksum(
                    self.w[l, f, t_hat, t, s]
                    for l in range(self.l_size)
                    for f in range(self.f_size)
                    for t_hat in range(self.t_size)
                    for t in range(self.growth_sets[l].loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat],
                           min(t_hat + self.parameters.max_periods_deployed, self.t_size))
                )
                for s in range(self.s_size)
            )
            , GRB.MAXIMIZE
        )

    def set_decomped_objective(self):
        if not self.MAB_shadow_prices_df.empty:
            self.model.setObjective(
                # This is the objective (5.2) - which represents the objective for biomass maximization
                gp.quicksum(
                    self.scenario.scenario_probabilities[s] * (
                    gp.quicksum(
                        self.w[l, f, t_hat, t, s] - #TODO: This was just changed to a +, check if it should be a -
                        self.x[l, f, t_hat, t, s] * self.MAB_shadow_prices_df.loc[(s, t)] if (s,t) in self.MAB_shadow_prices_df.index else 0.0
                        for l in range(self.l_size)
                        for f in range(self.f_size)
                        for t_hat in range(self.t_size)
                        for t in range(self.growth_sets[l].loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat],
                                       min(t_hat + self.parameters.max_periods_deployed, self.t_size))
                    ) -
                    gp.quicksum(
                        self.x[l, 0, t_hat, 60, s] * self.EOH_shadow_prices_df.loc[(s)] if (s) in self.EOH_shadow_prices_df.index else 0.0
                        for l in range(self.l_size)
                        for f in range(self.f_size)
                        for t_hat in range(self.t_size - self.parameters.max_periods_deployed + 1, self.t_size -1)
                    )
                    )
                    for s in range(self.s_size)
                )
                , GRB.MAXIMIZE
            )

        else:
            self.set_objective()

    def set_zero_ojective(self):
        self.model.setObjective(
            # This is the objective (5.2) - which represents the objective for biomass maximization
            gp.quicksum(
                self.scenario.scenario_probabilities[s] *
                gp.quicksum(
                    self.w[f, t_hat, t, s]
                    for f in range(self.f_size)
                    for t_hat in range(self.t_size)
                    for t in range(min(self.growth_sets.loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat], self.t_size - 1),
                           min(t_hat + self.parameters.max_periods_deployed, self.t_size - 1))
                )
                for s in range(self.s_size)
            )
            , GRB.MINIMIZE
        )


    def add_smolt_deployment_constraints(self):
        #FIXED
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
            gp.quicksum(self.y[l, f, t] for f in range(self.f_size)) <= self.parameters.smolt_deployment_upper_bound * self.deploy_bin[l, t]
            # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
            for t in range(1, self.t_size)
            for l in range(self.l_size)
        )

        #FIXED
        self.model.addConstrs(
            # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
            gp.quicksum(self.y[l, f, t] for f in range(self.f_size)) >= self.parameters.smolt_deployment_lower_bound * self.deploy_bin[l,t]
            # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
            for t in range(1, self.t_size)
            for l in range(self.l_size)
        )

        #FIXED
        self.model.addConstrs(  # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
            self.y[l, f, t] >= self.parameters.smolt_deployment_lower_bound * self.deploy_type_bin[l,f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
            for l in range(self.l_size)
        )

        #FIXED
        self.model.addConstrs(
            # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
            self.parameters.smolt_deployment_upper_bound * self.deploy_type_bin[l, f, t] >=  self.y[l,f, t]
            for t in range(1, self.t_size)
            for f in range(self.f_size)
            for l in range(self.l_size)
        )

        for l in range(self.l_size):
            if self.sites[l].init_biomass < 1:
                self.model.addConstr(
                    # This is the constraint (5.4) - which restricts the deployment of smolt to an upper bound, while forcing the binary deploy variable
                    gp.quicksum(self.y[l, f, 0] for f in range(self.f_size)) <= self.parameters.smolt_deployment_upper_bound * self.deploy_bin[l, 0]
                    ,name="Deploy limit if no init biomass at site"
                    # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
                )

                # FIXED
                self.model.addConstr(
                    # This is the constraint (5.4) - which restricts the deployment of smolt to a lower bound bound, while forcing the binary deploy variable
                    gp.quicksum(self.y[l, f, 0] for f in range(self.f_size)) >= self.parameters.smolt_deployment_lower_bound * self.deploy_bin[l, 0]
                    , name="Deploy limit if no init biomass at site"
                    # Divide by thousand, as smolt weight is given in grams, while deployed biomass is in kilos
                )

                # FIXED
                self.model.addConstrs(
                    # This is constraint (5.5) - setting a lower limit on smolt deployed from a single cohort
                    self.y[l, f, 0] >= self.parameters.smolt_deployment_lower_bound * self.deploy_type_bin[l, f, 0]
                    for f in range(self.f_size)


                )

                # FIXED
                self.model.addConstrs(
                    # This is constraint (Currently not in model) - setting an upper limit on smolt deployed in a single cohort #TODO: Add to mathematical model
                    self.parameters.smolt_deployment_upper_bound * self.deploy_type_bin[l, f, 0] >= self.y[l, f, 0]
                    for f in range(self.f_size)
                )


    def add_fallowing_constraints(self):
        #Fixed
        self.model.addConstrs(
            # This is the constraint (5.6) - It ensures that the required fallowing is done before the deploy variable can be set to 1
            self.parameters.min_fallowing_periods * self.deploy_bin[l, t] + gp.quicksum(
                self.employ_bin[l, tau, s] for tau in range(t - self.parameters.min_fallowing_periods, t))
            <= self.parameters.min_fallowing_periods
            for t in range(self.parameters.min_fallowing_periods, self.t_size)
            for s in range(self.s_size)
            for l in range(self.l_size)
        )

        #Fixed
        self.model.addConstrs(
            # This is an additional constraint - ensuring that only 1 deployment happens during the initial possible deployment period TODO: See if this needs to be implemented in the math model
            gp.quicksum(self.deploy_bin[l,t] for t in range(self.parameters.min_fallowing_periods)) <= 1
            for l in range(self.l_size)
        )

    def add_inactivity_constraints(self):
        #Fixed
        self.model.addConstrs(
            # This is the constraint (5.7) - ensuring that the site is not inactive longer than the max fallowing limit
            gp.quicksum(self.employ_bin[l, tau, s] for tau in range(t, min(t + self.parameters.max_fallowing_periods, self.t_size))) >= 1
            # The sum function and therefore the t set is not implemented exactly like in the mathematical model, but functionality is the same
            for s in range(self.s_size)
            for t in range(self.t_size)
            for l in range(self.l_size)
        )

    def add_harvesting_constraints(self):
        #Fixed
        self.model.addConstrs(
            # This is the first part of constraint (5.8) - which limits harvest in a single period to an upper limit
            gp.quicksum(self.w[l, f, t_hat, t, s] for f in range(self.f_size) for t_hat in range(t)) <= self.parameters.max_harvest *
            self.harvest_bin[l,t, s]
            for t in range(self.t_size)
            for s in range(self.s_size)
            for l in range(self.l_size)
        )

        #Fixed
        self.model.addConstrs(
            # This is the second part of constraint (5.8) - which limits harvest in a single period to a lower limit
            gp.quicksum(self.w[l, f, t_hat, t, s] for f in range(self.f_size) for t_hat in range(t)) >= self.parameters.min_harvest *
            self.harvest_bin[l, t, s]
            for t in range(self.t_size)
            for s in range(self.s_size)
            for l in range(self.l_size)
        )



    def add_biomass_development_constraints(self):
        self.model.addConstrs(  # This is constraint (5.9) - which ensures that biomass x = biomass deployed y
            self.x[l, f, t, t, s] == self.y[l,f, t]
            for f in range(self.f_size)
            for t in range(self.t_size)
            for s in range(self.s_size)
            for l in range(self.l_size)
        )

        #Fixed
        self.model.addConstrs(  # This represents the constraint (5.10) - which ensures biomass growth in the growth period
            self.x[l, f, t_hat, t + 1, s] == (1 - self.parameters.expected_production_loss) * self.x[l, f, t_hat, t, s] *
            self.growth_factors[l].loc[(self.smolt_weights[f], f"Scenario {s}", t_hat)][t]
            for l in range(self.l_size)
            for t_hat in range(self.t_size - 1)
            for f in range(self.f_size)
            for s in range(self.s_size)
            for t in
            range(min(t_hat, self.t_size), min(self.growth_sets[l].loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat], self.t_size))
        )

        #Fixed
        self.model.addConstrs(  # This is the constraint (5.11) - Which tracks the biomass employed in the harvest period
            self.x[l, f, t_hat, t + 1, s] == (1 - self.parameters.expected_production_loss) * self.x[l,f, t_hat, t, s] *
            self.growth_factors[l].loc[(self.smolt_weights[f], f"Scenario {s}", t_hat)][t] - self.w[l,f, t_hat, t, s]
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for f in range(self.f_size)
            for s in range(self.s_size)
            for t in range(min(self.growth_sets[l].loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat], self.t_size),
                           min(t_hat + self.parameters.max_periods_deployed, self.t_size))

        )


        self.model.addConstrs(
            # This is the constraint (5.12) - Which forces the binary employement variable to be positive if biomass is employed
            gp.quicksum(self.x[l, f, t_hat, t, s] for f in range(self.f_size)) <= self.employ_bin[l, t, s] * self.parameters.bigM
            for l in range(self.l_size)
            for s in range(self.s_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, min(t_hat + self.parameters.max_periods_deployed, self.t_size))
        )

    #Fixed
    def add_MAB_requirement_constraint(self):
        self.model.addConstrs(
            gp.quicksum(self.x[l, f, t_hat, t, s] for f in range(self.f_size)) <= self.sites[l].MAB_capacity
            for l in range(self.l_size)
            for t_hat in range(self.t_size)
            for t in range(t_hat, min(t_hat + self.parameters.max_periods_deployed, self.t_size +1))
            for s in range(self.s_size)
        )

    def add_MAB_company_requirement_constraint(self):
            for t in range(self.t_size + 1):
                for s in range(self.s_size):
                    self.model.addConstr(
                        gp.quicksum(self.x[l, f, t_hat, t, s] for l in range(self.l_size) for t_hat in range(min(t + 1, 60)) for f in range(self.f_size))
                         <= configs.MAB_COMPANY_LIMIT
                        , name="company MAB limit"
                    )

    def add_end_of_horizon_constraint(self):
            for s in range(self.s_size):
                self.model.addConstr(
                    gp.quicksum(self.x[l, f, t_hat, 60, s] for l in range(self.l_size) for t_hat in range(60 - self.parameters.max_periods_deployed, 59) for f in range(self.f_size))
                    >=
                    self.parameters.eoh_down_ratio *
                    gp.quicksum(self.y[l, f, 0] for l in range(self.l_size) for f in range(self.f_size))
                    , name="EOH down"
                )

            for s in range(self.s_size):
                self.model.addConstr(
                    gp.quicksum(self.x[l, f, t_hat, 60, s] for l in range(self.l_size) for t_hat in range(60 - self.parameters.max_periods_deployed, 59) for f in range(self.f_size))
                    >=
                    self.parameters.MAB_company_limit * self.parameters.MAB_util_end
                    , name="Second E0H down"
                )


    def add_initial_condition_constraint(self): #TODO: Add initial constraints
        for l in range(self.l_size):
            if self.sites[l].init_biomass > 1:
                self.model.addConstr(
                    self.y[l,0,0] == self.sites[l].init_biomass,

                    name="Initial Condition"
                )
                self.model.addConstr(
                    self.deploy_bin[l,0] == 1,
                    name="Initial Condition bin"
                )

    def add_forcing_constraints(self):
        self.model.addConstrs(
            # TODO:This is a forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[l,f, t_hat, t, s] == 0
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for s in range(self.s_size)
            for t in range(0, min(self.growth_sets[l].loc[(self.smolt_weights[f], f"Scenario {s}")][t_hat], self.t_size))
        )

        self.model.addConstrs(
            # TODO:This is a second forcing constraint that is not in the mathematical model, put it in the model somehow
            self.w[l, f, t_hat, t, s] == 0
            for l in range(self.l_size)
            for f in range(self.f_size)
            for t_hat in range(self.t_size)
            for t in range(min(t_hat + self.parameters.max_periods_deployed, self.t_size), self.t_size)
            for s in range(self.s_size)
        )

    def add_x_forcing_constraint(self):#TODO: check if used or remove
        self.model.addConstrs(
            self.x[l, f, t_hat, self.t_size, s] <= self.deploy_bin[l,t_hat] * self.parameters.MAB_company_limit
            for t_hat in range(self.t_size)
            for l in range(self.l_size)
            for f in range(self.f_size)
            for s in range(self.s_size)
        )

    def add_up_branching_constraints(self):
        for indice in self.branching_variable_indices_up:
            self.model.addConstr(
                self.deploy_bin[0, indice] == 1, #l set to 0 as this should only be used when there is only one site
                name = "Branching constraint"
            )


    def add_down_branching_constraints(self):
        for indice in self.branching_variable_indices_down:
            self.model.addConstr(
                self.deploy_bin[0, indice] == 0, #l set to 0 as this should only be used when there is only one site
                name = "Branching constraint"
            )

    def print_solution_to_excel(self):
        if self.model.status != GRB.INFEASIBLE:
            i = 0

            consolidted_df = pd.concat([df for df in self.get_second_stage_variables_df()], keys=[i for i in range(self.l_size)])


            path = configs.OUTPUT_DIR + f"/monolithic_model_scenario{configs.NUM_SCENARIOS}_sites{configs.NUM_LOCATIONS}.xlsx"
            consolidted_df.to_excel(path)
            i+=1

    def print_solution_to_csv(self):
        if self.model.status != GRB.INFEASIBLE:
            export_df = pd.concat(self.get_second_stage_variables_df(), keys=[i for i in range(self.l_size)])
            export_df.index.names = self.column_index_names
            export_df.to_csv(f"./init_columns/column_s{self.s_size}_l{self.l_size}.csv")

    def get_solution_as_df(self):
        if self.model.status != GRB.INFEASIBLE:
            export_df = pd.concat(self.get_second_stage_variables_df(), keys=[i for i in range(self.l_size)])
            export_df.index.names = self.column_index_names
            return export_df

    def get_solution_df_from_csv(self, path):
        df = pd.read_csv(path, index_col=self.column_index_names)
        return df


    def plot_solutions_x_values_per_site(self):
        """
        EXPLANATION: This function plots the x values, i.e the biomass at the location for every period in the planning horizion
        :return:
        """
        #Fetching the solution dataframe

        dfs = self.get_second_stage_variables_df()
        scenarios = self.s_size
        #Declaring a list for storing the x values


        #Iterating through the dataframe to sum together the biomass at location for every site
        for l in range(self.l_size):
            x_values = []
            df = dfs[l]
            for s in range(scenarios):
                scenarios_x_values = []
                for t in range(self.t_size +1):
                    x_t = 0
                    for t_hat in range(self.t_size):
                        for f in range(self.f_size):
                            x_t += df.loc[(s, f, t_hat, t)]["X"] if (s, f, t_hat, t) in df.index else 0.0

                    scenarios_x_values.append(x_t)
                x_values.append(scenarios_x_values)

            #Declaring a variable for storing the x-axis to be used in the plot
            x_axis = np.arange(0,self.t_size + 1)

            #Adding to the plots
            for scenario in x_values:
                plt.plot(x_axis, scenario)

            plt.title(f"Biomass at site {self.sites[l].name} iteration {self.iterations}")
            plt.ylabel("Biomass")
            plt.xlabel("Periods")
            path = f'results/plots/{self.sites[l].name}{self.iterations}.png'
            plt.savefig(path)

            plt.show()
            plt.clf()
            plt.close()

    def plot_solutions_x_values_aggregated(self):
        """
               EXPLANATION: This function plots the x values, i.e the biomass at the location for every period in the planning horizion
               :return:
               """
        # Fetching the solution dataframe
        dfs = self.get_second_stage_variables_df()
        scenarios = self.s_size

        # Declaring a list for storing the x values
        x_values = []

        # Iterating through the dataframe to sum together the biomass at location for every site
        for l in range(self.l_size):
            df = dfs[l]
            for s in range(scenarios):
                scenarios_x_values = []
                for t in range(self.t_size + 1):
                    x_t = 0
                    for t_hat in range(self.t_size):
                        for f in range(self.f_size):
                            x_t += df.loc[(s, f, t_hat, t)]["X"] if (s, f, t_hat, t) in df.index else 0.0

                    scenarios_x_values.append(x_t)
                x_values.append(scenarios_x_values)

            # Declaring a variable for storing the x-axis to be used in the plot
            x_axis = np.arange(0, self.t_size + 1)

            # Adding to the plots
            for scenario in x_values:
                plt.plot(x_axis, scenario)

            plt.title(f"Biomass at site aggregated")
            plt.ylabel("Biomass")
            plt.xlabel("Periods")
            path = configs.OUTPUT_DIR + f'/plot_aggregate.png'
            plt.savefig(path)

        plt.show()

    def plot_solutions_agregate_vs_MAB_limit(self):
        dfs = self.get_second_stage_variables_df()
        scenarios = self.s_size

        # Declaring a list for storing the x values
        x_values = []

        # Iterating through the dataframe to sum together the biomass at location for every site
        for s in range(scenarios):
            scenarios_x_values = []
            for t in range(self.t_size + 1):
                x_t_aggregate = 0
                for t_hat in range(self.t_size):
                    for f in range(self.f_size):
                        for l in range(self.l_size):
                            df = dfs[l]
                            x_t_aggregate += df.loc[(s, f, t_hat, t)]["X"] if (s, f, t_hat, t) in df.index else 0.0



                scenarios_x_values.append(x_t_aggregate)

            x_values.append(scenarios_x_values)

            # Declaring a variable for storing the x-axis to be used in the plot
        x_axis = np.arange(0, self.t_size + 1)

            # Adding to the plots
        for scenario in x_values:
            plt.plot(x_axis, scenario)

        plt.plot(x_axis, np.full(self.t_size + 1, self.parameters.MAB_company_limit))
        plt.ylim(0, self.parameters.MAB_company_limit * 1.20)

        plt.title(f"Biomass at site aggregated")
        plt.ylabel("Biomass")
        plt.xlabel("Periods")

        path = configs.OUTPUT_DIR + f'/MAB_utilization.png'
        plt.savefig(path)


        plt.show()

    def get_deploy_period_list(self):
        deploy_periods_list = []
        for l in range(self.l_size):
            deploy_period_list_l = [i for i in range(0, self.t_size) if self.deploy_bin[l,i].x == 1]
            deploy_periods_list.append(deploy_period_list_l)
        return deploy_periods_list

    def get_deploy_period_list_per_cohort(self):
        data_storage =[]
        for f in range(self.f_size):
            deploy_period_list_f = [t for t in range(0, self.t_size) if self.deploy_type_bin[f,t].x == 1]
            data_storage.append(deploy_period_list_f)

        df = pd.DataFrame(data_storage, index=[f"Smolt type {f}" for f in range(self.f_size)])
        return df

    def get_deploy_amounts_df(self):
        deploy_periods_list = self.get_deploy_period_list()

        data_storage = []
        for f in range(self.f_size):
            deploy_amounts_f = [self.y[f,t].x for t in deploy_periods_list]
            data_storage.append(deploy_amounts_f)

        df = pd.DataFrame(data_storage, index=[f"Smolt type {f}" for f in range(self.f_size)], columns=deploy_periods_list)
        return df

    def get_second_stage_variables_df(self):
        """
        This is the for loop from hell.
        It iterates through all non-zero variables and writes the variables for X, W, Employ_bin and Harvest_bin to a ginormous dataframe
        Please don't touch it
        """


        deploy_periods_list = self.get_deploy_period_list()

        deploy_period_dfs = []
        for l in range(self.l_size):
            deploy_period_list = deploy_periods_list[l]
            df_storage = []
            for s in range(self.s_size):
                l1_df_storage = []
                for f in range(self.f_size):
                    l2_df_storage = []
                    for deploy_period in deploy_period_list:
                        l3_data_storage = []
                        for t in range(deploy_period, min(deploy_period + self.parameters.max_periods_deployed, self.t_size)):
                            x_entry = round(self.x[l, f, deploy_period, t, s].x,2)
                            w_entry = round(self.w[l, f, deploy_period, t, s].x, 2)
                            employ_entry = self.employ_bin[l, t, s].x
                            harvest_entry = self.harvest_bin[l, t, s].x
                            deploy_entry = self.deploy_bin[l, t].x
                            deploy_type_entry = self.deploy_type_bin[l, f, t].x
                            y_entry = round(self.y[l, f, t].x,2)
                            l3_data_storage.append(
                                [x_entry, w_entry, employ_entry, harvest_entry, deploy_entry, deploy_type_entry,
                                 y_entry])
                        if self.x[l, f, deploy_period, 60, s].x > 0:
                            l3_data_storage.append([self.x[l, f, deploy_period, 60, s].x, 0, 0, 0, 0, 0, 0])
                        columns = ["X", "W", "Employ bin", "Harvest bin", "Deploy bin", "Deploy type bin", "Y"]
                        index = [i + deploy_period for i in range(len(l3_data_storage))]
                        l2_df_storage.append(pd.DataFrame(l3_data_storage, columns=columns, index=index))
                    keys_l2 = [i for i in deploy_period_list]
                    l1_df_storage.append(pd.concat(l2_df_storage, keys=keys_l2))
                keys_l1 = [i for i in range(len(l1_df_storage))]
                df_storage.append(pd.concat(l1_df_storage, keys=keys_l1))
            keys = [i for i in range(len(df_storage))]
            df = pd.concat(df_storage, keys=keys)

            df.index.names = self.solution_index_names

            deploy_period_dfs.append(df)

        return deploy_period_dfs


    def set_shadow_prices_df(self, shadow_prices_df_mab, shadow_prices_df_EOH):
        self.MAB_shadow_prices_df = shadow_prices_df_mab
        self.EOH_shadow_prices_df = shadow_prices_df_EOH

    def write_objective_value_to_file(self, time):
        file_path = configs.OUTPUT_DIR + f"/monolithic_results_s{configs.NUM_SCENARIOS}_l{configs.NUM_LOCATIONS}.txt"
        with open(file_path, "a") as file:
            if self.model.status == GRB.OPTIMAL:
                file.write(f"OBJECTIVE:{self.model.ObjVal}")
                file.write(f";duration:{time}")






