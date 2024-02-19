import pandas as pd
import numpy as np
import configs
import input_data


class Scenarios:
    """
    EXPLANATION:
    This is a class that contains the temperature distributions for each scenario at each site, for the planning period
    It also contains the necessary methods for calculating the temperatures for every scenario, and inputs for what variations we shall see
    """

    #Variables to be read
    scenario_temperatures_per_site_df = None

    #Variables to be input here:
    scenarios_variations = configs.SCENARIOS_VARIATIONS
    scenario_probabilities = configs.SCENARIO_PROBABILITIES
    num_scenarios = len(scenarios_variations)

    def __init__(self, base_temperatures_df):
        self.scenario_temperatures_per_site_df = self.generate_scenarios_df(base_temperatures_df)


    def generate_scenarios_df(self, base_temperature_df):
        """
        This function takes as an input the observed temperatures at all locations, and generates possible scenarios for every site based on the scenario variations input.
        It then returns the calculated scenarios in a pandas multi index dataframe.

        :param base_temperature_df: A dataframe containing the observed temperatures for 1 year at each site for the company
        :return: scenario_temperatures_per_site_df - a multi-index dataframe containing temperatures for every scenario that is generated
        """
        #Declaring variables for code to be more readable
        scenario_temperatures = []                              #A list for containing dataframes of scenario temperatures for each site, later to be put into the concatenated dataframe
        sites = base_temperature_df.index                       #A list containing the name of all sites, later to be used as indexes for the concatenated dataframe

        #Iterating through all sites
        for i in range(sites.size):
            #Generating a Dataframe containing all scenario temperatures for that site
            site_df = pd.DataFrame([base_temperature_df.iloc[i] * self.scenarios_variations[j] for j in range(self.num_scenarios)], index=[f"Scenario {j}" for j in range(self.num_scenarios)])
            #Appending the site with scenario temperatures to the list storig thedataframes
            scenario_temperatures.append(site_df)

        #Concatinating the dataframes for all sites into on multiindex dataframe
        scenario_temperatures_per_site_df = pd.concat([df for df in scenario_temperatures], keys=sites)

        return scenario_temperatures_per_site_df

if __name__ == '__main__':
    input_data = input_data.InputData()
    scenarios = Scenarios(input_data.temperatures_df)

    scenarios_temperatures = scenarios.scenario_temperatures_per_site_df
    scenarios_temperatures.to_excel("scenarios_temperatures.xlsx")