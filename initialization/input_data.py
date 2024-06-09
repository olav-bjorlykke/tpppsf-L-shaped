import pandas as pd
import numpy as np
import random
import initialization.parameters as parameters
import initialization.configs as configs

class InputData:
    """
    The input Data class reads in and stores the input data in pandas Dataframes, making it more accessible for use by other modules
    """
    #Variables for storing the read data
    temperatures_df = None
    TGC_df = None
    mortality_rates_df = None
    #Variables to be read
    scenario_temperatures_per_site_df = None

    #Variables to be input here:
    
        
    def __init__(self, configs):
        self.configs = configs
        scenarios_variations = configs.SCENARIOS_VARIATIONS
        self.scenario_probabilities = configs.SCENARIO_PROBABILITIES
        self.num_scenarios = len(scenarios_variations)
        #This is constructor function, it reads in data to the variables by calling the read methods
        self.temperatures_df = self.read_input_temperatures("data/temps_aasen.txt")
        self.TGC_df = self.read_input_tgc("data/tgc_aasen.txt")
        self.mortality_rates_df = self.read_input_mortality_rates("data/mortality_aasen_new.txt")
        self.scenario_temperatures_per_site_df = self.generate_scenarios_df(self.temperatures_df)
    """
    Defining all methods for reading in data from files. Due to the data files being of different format a method is created for every file. 
    """
    def read_input_temperatures(self,filepath):
        """
        EXPLANATION:
        This function takes in a txt file, containing the temperatures measures at the sites in over a twelve month period.
        This function parses the data into a pandas dataframe for usability, and extends the data to be for 5 years or 60 months.

        :param filepath: The filepath of the stored temperature data
        :return: temperatures_df - a dataframe containing temperatures for every month in a 5 year planning horizon for all sites.
        """
        #Open and read file
        file = open(filepath, "r")
        data = file.readlines()
        for i in range(len(data)):
            #Parse data in file
            data[i] = data[i].split("(")
            data[i][0] = data[i][0].strip().split(" ")
            data[i][1] = data[i][1].strip()

        # Stores temperature data in variable for readability and extends the data from 12 months to 60 months, by repeating the array 5 times
        temperatures_entire_horizon = [np.tile(elem[0],5) for elem in data]

        # Read data into dataframe
        temperatures_df = pd.DataFrame([temp_site_array for temp_site_array in temperatures_entire_horizon],  columns=[i for i in range(len(temperatures_entire_horizon[0]))], index=[elem[1] for elem in data])

        #Turn every instance of string into a float
        temperatures_df = temperatures_df.map(lambda x: float(x) if isinstance(x, str) else x)
        return temperatures_df

    def read_input_tgc(self,filepath):
        """
        Reads in data from a file, and puts it into a more usable pandas dataframe

        :param filepath: A txt file containg TGC numbers
        :return: df - a dataframe containing tgc for different smolt types across a deploy period
        """

        #Open and read file
        file = open(filepath,"r")
        data = file.readlines()
        for i in range(len(data)):
            #Parse file
            data[i] = data[i].split("(")
            data[i][0] = data[i][0].strip().split(",")
            data[i][1] = data[i][1].strip()
        #Read data into dataframe
        df = pd.DataFrame([elem[0] for elem in data], columns=[i for i in range(len(data[0][0]))], index=[elem[1] for elem in data])

        # Turn every instance of string into a float
        df = df.map(lambda x: float(x) if isinstance(x, str) else x)
        return df

    def read_input_mortality_rates(self,filepath):
        """
        Reads in mortality rates from a txt file, and puts it into a more useable pandas dataframe.

        :param filepath: The filepath to a txt file, conaining mortality rate data
        :return: df - a pandas dataframe contraining mortality rates for all smolt types
        """
        #Read data from file
        file = open(filepath, "r")
        data = file.readlines()
        for i in range(len(data)):
            #parse data
            data[i] = data[i].strip().split(",")

        #Define variables for input into dataframe
        indices = [elem[0] for elem in data]
        rates = [elem[1:] for elem in data]

        #Read data into dataframe
        df = pd.DataFrame(rates,index=indices, columns=[i for i in range(len(rates[0]))])

        # Turn every instance of string into a float
        df = df.map(lambda x: float(x) if isinstance(x, str) else x)
        return df

    def generate_scenarios_df(self, base_temperature_df):
        """
        This function takes as an input the observed temperatures at all locations, and generates possible scenarios for every site based on the scenario variations input.
        It then returns the calculated scenarios in a pandas multi index dataframe.

        :param base_temperature_df: A dataframe containing the observed temperatures for 1 year at each site for the company
        :return: scenario_temperatures_per_site_df - a multi-index dataframe containing temperatures for every scenario that is generated
        """
        #Declaring variables for code to be more readable
        if self.configs.RANDOM_SCENARIOS:
            random.seed(random.random())
        else:
            random.seed(parameters.random_seed)

        scenario_temperatures = []                              #A list for containing dataframes of scenario temperatures for each site, later to be put into the concatenated dataframe
        sites = base_temperature_df.index                       #A list containing the name of all sites, later to be used as indexes for the concatenated dataframe

        #Iterating through all sites
        if self.configs.AVERAGE_VALUES:
            for i in range(sites.size):
                #Generating a Dataframe containing all scenario temperatures for that site
                #site_df = pd.DataFrame([base_temperature_df.iloc[i] * self.scenarios_variations[j] for j in range(self.num_scenarios)], index=[f"Scenario {j}" for j in range(self.num_scenarios)])
                base_temps = base_temperature_df.iloc[i]
                scenarios_temps_list = []
                for j in range(self.configs.NUM_SCENARIOS):

                    scenario_temps = [base_temps[t] for t in range(len(base_temps))]
                    scenarios_temps_list.append(scenario_temps)


                site_df = pd.DataFrame(scenarios_temps_list, index=[f"Scenario {j}" for j in range(self.num_scenarios)])
                #Appending the site with scenario temperatures to the list storig thedataframes
                scenario_temperatures.append(site_df)

        else:
            for i in range(sites.size):
                # Generating a Dataframe containing all scenario temperatures for that site
                # site_df = pd.DataFrame([base_temperature_df.iloc[i] * self.scenarios_variations[j] for j in range(self.num_scenarios)], index=[f"Scenario {j}" for j in range(self.num_scenarios)])
                base_temps = base_temperature_df.iloc[i]
                scenarios_temps_list = []
                for j in range(self.configs.NUM_SCENARIOS):
                    scenario_temps = [base_temps[t] * random.uniform(0.90, 1.05) for t in range(len(base_temps))]
                    scenarios_temps_list.append(scenario_temps)

                site_df = pd.DataFrame(scenarios_temps_list, index=[f"Scenario {j}" for j in range(self.num_scenarios)])
                # Appending the site with scenario temperatures to the list storig thedataframes
                scenario_temperatures.append(site_df)

        #Concatinating the dataframes for all sites into on multiindex dataframe
        scenario_temperatures_per_site_df = pd.concat([df for df in scenario_temperatures], keys=sites)

        return scenario_temperatures_per_site_df

