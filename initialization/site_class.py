import numpy as np
import pandas as pd
import initialization.parameters as parameters
from initialization.input_data import InputData


class Site:
    """
    EXPLANATION OF CLASS:
    This class shall contain all the necessary data that is site specific, to solve the tactical production planning problem for a single site
    """
    #Callable parameters
    growth_sets = None                                      #A datafrane containing the set of periods where harvest is not allowed, following release for all scenarios and smolt weights
    max_periods_deployed = None                             #Max number of periods a cohort can be deployed
    MAB_capacity = None                                     #Max biomass capacity at the site
    init_biomass_at_site = False
    init_biomass = 0                                        #Biomass at the site at the start of the planning period
    init_avg_weight = 0                                     #The average weight of a salmon if the site has biomass deployed
    num_months_deployed = 0                                 #The number of months the cohort has been deployed, if there is biomass at the site
    growth_per_scenario_df = None                           #A dataframe containing the growth factor for every period, scenario, smolt weight, deploy period combination
    weight_dev_per_scenario_df = None                       #A dataframe containing the weight development for every period, scenario, smolt weight, deploy period combination
    name = None

    #Parameters for calculations within this class
    TGC_array = None                                        #An array containing the growth coefficients for a given smolt type
    scenario_temps = None                                   #An array containing the temperatures for all possible scenarios

    def __init__(self,
                 scenario_temperatures,
                 MAB_capacity,
                 configs,
                 site_name = "Not Set",
                 init_biomass = 0,
                 init_avg_weight = 0,
                 init_biomass_months_deployed=0,
                 ):

        #Setting class variables
        self.configs = configs
        self.TGC_array = InputData(self.configs).TGC_df.iloc[0]                                               #Array of all TGC for a possible deploy period
        self.MAB_capacity = MAB_capacity                                          #Max biomass capacity at a single site
        self.init_biomass = init_biomass                                          #Initial biomass at the site, i.e biomass in the first period
        self.init_avg_weight = init_avg_weight
        self.num_months_deployed = init_biomass_months_deployed
        self.smolt_weights = parameters.smolt_weights                                      #Array of possible smolt weights
        self.scenario_temps = scenario_temperatures                               #Array of scenario temperatures for the site
        self.max_periods_deployed = len(self.TGC_array)                                #The maximum number of periods a cohort can be deployed
        self.name = site_name                                                     #The name of the site

        #Setting the init biomass at site variable
        if init_biomass != 0: self.init_biomass_at_site = True

        print(f"calculating at site {self.name}")
        #Calulating growth and weight development dataframes for all scenarios and possible smolt weights
        self.growth_per_scenario_df = self.calculate_growth_df_for_scenarios_and_smolt_weights(self.smolt_weights, scenario_temperatures)
        self.weight_dev_per_scenario_df = self.calculate_weight_df_for_scenarios_and_smolt_weights(self.smolt_weights, scenario_temperatures)

        #Calculating the growth sets
        self.growth_sets = self.calculate_growth_sets_from_weight_dev_df(self.weight_dev_per_scenario_df, parameters.weight_req_for_harvest)


    def calculate_growth_sets_from_weight_dev_df(self, weight_development_df, weight_req_for_harvest):
        """
        EXPLANATION:
        A function that calculates the growth sets. A growth is the set of periods following deployment of smolt where harvest is not allowed.
        For instance if smolt is relead in period t, harvest would not be allowed until period t+12,
        where period t + 12 is the first period where the expected weight is more than the weight requirement for harvest.

        The growth set is represented by a list where the first entry is the release period and the second entry is the last first period where harvest is allowed.
        For instance [12,24]. Indicating that smolt released in period 12, can not be harvested until period 24.

        :param weight_development_df: A dataframe containing the expected weight developments for all deploy periods, scenarios and smolt weights
        :param weight_req_for_harvest: A float, indicating the minimum weight before harvest is allowed
        :return: growth_sets_df - a dataframe containing growth sets for all periods, deploy periods, scenarios and smolt weights
        """

        #Creating List for storing data to be put into dataframe
        data_storage = []
        #Creating variables to store the indexes
        weights = weight_development_df.index.get_level_values("weight").unique()
        scenarios = weight_development_df.index.get_level_values("scenario").unique()

        #Iterating over all possible weights
        for weight in weights:
            #Creating variable for storing data to be put into dataframe
            scenario_growth_sets = []
            for i, scenario in enumerate(scenarios):
                #Calculating the growth sets for a single scenario and smolt weight:
                growth_sets = self.calculate_growth_sets_for_single_scenario_and_weight(weight_development_df.loc[(weight,scenario)], weight_req_for_harvest)
                scenario_growth_sets.append(growth_sets)

            #Creating a dataframe containing the growth sets for every scenario for a given smolt weight
            scenario_df = pd.DataFrame(scenario_growth_sets, index=scenarios)
            data_storage.append(scenario_df)

        #Creating a multi-index dataframe containing all growth sets for all scenarios and smolt weights
        growth_sets_df = pd.concat(data_storage, keys=weights)

        return growth_sets_df

    def calculate_growth_sets_for_single_scenario_and_weight(self, weight_dev_df_single_scenario_and_weight, weight_req_for_harvest):
        """
        EXPLANATION: calculates the growth set for a single scenario and smolt weight. See "calculate_growth_sets_from_weight_dev_df" for definition of growth set.

        :param weight_dev_df_single_scenario_and_weight: A dataframe containing the expected weight developments for a single scenario and smolt weight
        :param weight_req_for_harvest: The reuired weight for harvest to be allowed
        :return: growth_set_array - an array containing growth sets for every deploy period
        """
        #Creating array
        growth_set_array = []
        #Iterating through every deploy period
        for i in range(weight_dev_df_single_scenario_and_weight.index.size):
            #Iterating through every growth period - currently iterating through all periods, but result is the same
            for j in range(len(weight_dev_df_single_scenario_and_weight.iloc[i])):
                #If the expected weight is above the weight limit for period j, then the growth set is found and added to the growth_set_array
                if weight_dev_df_single_scenario_and_weight.iloc[i][j] > weight_req_for_harvest:
                    growth_set_array.append(j)
                    break
                #If harvest is not allowed until after the end of the planning horizon, we set the growth set to be the maximum + 1
                elif j == len(weight_dev_df_single_scenario_and_weight.iloc[i]) -1:
                    growth_set_array.append(60)

        return growth_set_array

    def calculate_weight_df_for_scenarios_and_smolt_weights(self, smolt_weights, scenarios_temperatures):
        """
               EXPLANATION: Calculates the weight development, for all smolt weights, for very scenario, release period and subsequent growth periods.

               :param smolt_weights: An array containing all possible smolt weights for release
               :param scenarios_temperatures: A dataframe containing temperaturs for all scenarios
               :return: growth_frame_all_scenarios_and_weights - a 4-D dataframe with the data described in the explanation.
               """
        #TODO: Add detailed comments
        data_storage = []
        for weight in smolt_weights:
            scenario_weight_frames_given_weight = self.calculate_weight_df_for_scenarios(weight, scenarios_temperatures)
            data_storage.append(scenario_weight_frames_given_weight)

        weight_frame_all_scenarios_and_weights = pd.concat(data_storage, keys=smolt_weights)

        #Naming the indexes for easier access
        weight_frame_all_scenarios_and_weights.index.names = ["weight", "scenario", "period"]
        return weight_frame_all_scenarios_and_weights

    def calculate_weight_df_for_scenarios(self, smolt_weight, scenarios_temperatures):
        """
                EXPLANATION: Calculates the expected weight for every period, following every possible release in every scenario for a given smolt weight.

                :param smolt_weight: The weight of the smolt at release
                :param scenarios_temperatures: a dataframe containing the temperatures for all scenarios across the planning period
                :return: growth_frame_all_scenarios - a 3-D dataframe containing the growth factor for every period, following every release and every scenario.
                """
        #TODO: Add detailed comments
        data_storage = []
        for i in range(len(scenarios_temperatures.index)):
            temp_array = scenarios_temperatures.iloc[i]
            scenario_weight_frame = self.calculate_weight_df(temp_array, smolt_weight)
            data_storage.append(scenario_weight_frame)

        weight_frame_all_scenarios = pd.concat(data_storage, keys=scenarios_temperatures.index)
        return weight_frame_all_scenarios

    def calculate_growth_df_for_scenarios_and_smolt_weights(self, smolt_weights, scenarios_temperatures):
        """
        EXPLANATION: Calculates the growth factor, for all smolt weights, for very scenario, release period and subsequent growth periods.

        :param smolt_weights: An array containing all possible smolt weights for release
        :param scenarios_temperatures: A dataframe containing temperaturs for all scenarios
        :return: growth_frame_all_scenarios_and_weights - a 4-D dataframe with the data described in the explanation.
        """
        #TODO: Add detailed comments

        data_storage = []
        for weight in smolt_weights:
            scenario_growth_frames_given_weight = self.calculate_growth_df_for_scenarios(weight, scenarios_temperatures)
            data_storage.append(scenario_growth_frames_given_weight)

        growth_frame_all_scenarios_and_weights = pd.concat(data_storage, keys=smolt_weights)
        return growth_frame_all_scenarios_and_weights

    def calculate_growth_df_for_scenarios(self, smolt_weight, scenarios_temperatures):
        """
        EXPLANATION: Calculates the growth factor for every period, following every possible release in every scenario for a given smolt weight.

        :param smolt_weight: The weight of the smolt at release
        :param scenarios_temperatures: a dataframe containing the temperatures for all scenarios across the planning period
        :return: growth_frame_all_scenarios - a 3-D dataframe containing the growth factor for every period, following every release and every scenario.
        """
        #TODO: Add detailed comments

        data_storage = []
        for i in range(len(scenarios_temperatures.index)):
            temp_array = scenarios_temperatures.iloc[i]
            scenario_growth_frame = self.calculate_growth_df_from_weight_df(self.calculate_weight_df(temp_array, smolt_weight))
            data_storage.append(scenario_growth_frame)



        growth_frame_all_scenarios = pd.concat(data_storage, keys=scenarios_temperatures.index)
        return growth_frame_all_scenarios

    def calculate_growth_factor(self, weight, TGC, temperature, duration): #TODO:put into its own class or set of functions
        """
        A function for calculating the growth factor for one period
        :param weight: the weight of an individual salmon in the growth period
        :param TGC: The growth coefficient
        :param temperature: The average sea temperature in the period
        :param duration: The duration of the period in days
        :return: A float G which is the growth factor
        """
        new_weight = (weight**(1/3) + (1/1000)*TGC*temperature*duration)**(3)
        G = new_weight/weight
        return G

    def calculate_weight_development(self, weight, TGC, temperature, duration=30): #TODO: put into its own class or set of fucntions along with growth factor
        """
        A function calculating the weight of a single salmon in period t+1, given the weight in period t. The calculation is based on Aasen(2021) and Thorarensen & Farrel (2011)
        :param weight: weight in period t
        :param TGC: the growth coefficient in period t
        :param temperature: the average sea temperature in period t
        :param duration: the duration of period t
        :return: new_weight, a float that defines the expected weight in period t+1
        """
        new_weight = (weight**(1/3) + TGC*temperature*duration/1000)**(3)
        return new_weight

    def calculate_weight_df(self, temp_array, smolt_weight):
        """
        A function that calculates the expected weigh development for all possible release periods in a planning horizon
        :param smolt_weight: The deploy weight of a Salmon
        :return: weight_df  -  an array containing expected weight development for every possible release period and subsequent growth and harvest periods
        """
        #Defining a weight array to contain the calculated data
        number_periods = len(temp_array)
        weight_array = np.zeros((number_periods, number_periods))
        for i in range(number_periods):
            #Iterate through all possible release periods i and set the initial weight
            weight_array[i][i] = smolt_weight
            for j in range(i, min(number_periods - 1, i + self.max_periods_deployed)):
                #Iterate through all possible growth periods j given release period i
                #Calculate the weight for period j+1 given the weight in period j using weight_growth() function and put into the array
                weight = weight_array[i][j]
                weight_array[i][j + 1] = self.calculate_weight_development(weight, temperature=temp_array[j], TGC=self.TGC_array[j - i])

        if self.init_biomass_at_site:
            weight_array[0] = self.calculate_weight_dev_of_initial_biomass(init_avg_weight=self.init_avg_weight, months_deployed=self.num_months_deployed, temp_array=temp_array)

        # Read data from array into dataframe
        weight_df = pd.DataFrame(weight_array, columns=[i for i in range(number_periods)], index=[i for i in range(number_periods)])

        return weight_df

    def calculate_growth_df_from_weight_df(self, weight_frame):
        """
        A function that calculates a growth factor for every possible relase and subsequent growth and harvest period. The growth factor indicates the growth from a period t to a period t+1
        :param weight_frame: The weight frame is a dataframe containing all possible weight developments.
        :return: growth_df  -  A dataframe with all the growth factors for every possible period
        """
        #Getting the number of periods from the growth array
        number_periods = weight_frame[0].size
        #Declare a growth_factor array containing all calculated growth factors
        growth_array = np.zeros((number_periods, number_periods))
        for i in range(number_periods):
            #Iterate through all possible release periods i
            for j in range(i, min(number_periods - 1, i + self.max_periods_deployed)):
                # Iterate through all possible growth periods j given release period i
                # Calculate the growth factor, using expected weight developments from the weight fram and input into array
                # Checking if the current period has biomass deployed -> putting in an if / else statement to avoid divide by zero errors
                if weight_frame.iloc[i][j] > 0:
                    growth_array[i][j] = weight_frame.iloc[i][j+1]/weight_frame.iloc[i][j]
                else:
                    growth_array[i][j] = 0

                growth_array[i][59] = growth_array[i][58]
        #Read data from array into dataframe
        growth_df = pd.DataFrame(growth_array, columns=[i for i in range(number_periods)],index=[i for i in range(number_periods)])
        return growth_df

    def calculate_weight_dev_of_initial_biomass(self, init_avg_weight, months_deployed, temp_array):
        #Setting the number of periods for easier use
        number_periods = len(temp_array)

        #Declaring an array to contain the weight development of the initial biomass
        weight_dev_array = np.zeros(number_periods)
        #Setting the weight in period 0 to be the initial average weight
        weight_dev_array[0] = init_avg_weight

        #Iterating through all periods, calculating the weight in next period based on the weight in the current period
        for i in range(0, self.max_periods_deployed - months_deployed):
            current_weight = weight_dev_array[i]
            weight_in_next_period = self.calculate_weight_development(current_weight, temperature=temp_array[i], TGC=self.TGC_array[i + months_deployed])
            weight_dev_array[i+1] = weight_in_next_period

        return weight_dev_array







