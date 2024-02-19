import numpy as np
import pandas as pd
from input_data import InputData
from site_class import Site
from scenarios import Scenarios
from model import Model
import configs

#Fetching temperature data from det dataclasses
input_data =InputData()
scenarios_data = Scenarios(input_data.temperatures_df)

#Declaring the string variables for each area here to avoid errors and duplication of efforts
area_vesteralen_string = "Vesteralen"
area_nordtroms_string = "Nord-Troms"
area_senja_string = "Senja"

"""
DECLARING THE PARAMETERS FOR ALL SITES
"""
site_1 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120 * 1000,
    init_biomass=907 * 1000,
    init_avg_weight=2410,
    init_biomass_months_deployed=8,
    site_name="INNERBROKLØYSA"
)

site_2 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=2340 * 1000,
    site_name="SANDAN SØ"
)

site_3 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120*1000,
    init_biomass=378 * 1000,
    init_avg_weight=695,
    init_biomass_months_deployed=4,
    site_name="TROLLØYA SV"
)

site_4 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120*1000,
    site_name="KUNESET"
)

site_5 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120*1000,
    init_biomass=661 * 1000,
    init_avg_weight=1634,
    init_biomass_months_deployed=6,
    site_name="STRETARNESET"
)
site_6 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120*1000,
    site_name="DALJORDA"
)
site_7 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=5500*1000,
    site_name="REINSNESØYA"
)
site_8 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3120*1000,
    init_biomass=661 * 1000,
    init_avg_weight=1634,
    init_biomass_months_deployed=6,
    site_name="LANGHOLMEN N"
)
site_9 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=3900*1000,
    site_name="BREMNESØYA"
)
site_10 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_vesteralen_string],
    MAB_capacity=2340*1000,
    init_biomass=1312 * 1000,
    init_avg_weight=2458,
    init_biomass_months_deployed=12,
    site_name="HOLAND"
)
site_11 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_senja_string],
    MAB_capacity=4680*1000,
    init_biomass=4296 * 1000,
    init_avg_weight=5464,
    init_biomass_months_deployed=17,
    site_name="LAVIKA"
)

site_12 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_senja_string],
    MAB_capacity=3600*1000,
    init_biomass=3536 * 1000,
    init_avg_weight=6536,
    init_biomass_months_deployed=18,
    site_name="FLESEN"
)
site_13 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_senja_string],
    MAB_capacity=3900*1000,
    site_name="KVENBUKTA V"
)
site_14 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_nordtroms_string],
    MAB_capacity=3600*1000,
    site_name="HAGEBERGAN"
)
site_15 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_nordtroms_string],
    MAB_capacity=3500*1000,
    init_biomass=961 * 1000,
    init_avg_weight=1411,
    init_biomass_months_deployed=7,
    site_name="RUSSELVA"
)
site_16 = Site(
    scenario_temperatures=scenarios_data.scenario_temperatures_per_site_df.loc[area_nordtroms_string],
    MAB_capacity=5000*1000,
    site_name="HAUKØYA Ø",
)

"""
DECLARING LISTS OF SITES AND SUBPROBLEMS
The different lists are used for running different variations of the planning problem. 
"""

#Defining the list containing all sites
large_sites_list = [site_1, site_2, site_3, site_4, site_5, site_6,
                    site_7, site_8, site_9, site_10, site_11, site_12,
                    site_13, site_14, site_15, site_16]

#Defining the list containing 8 sites
medium_sites_list = [site_3, site_4, site_6, site_7, site_8,
                site_9, site_10, site_15]

#Defining the list containing 3 sites
short_sites_list = [site_4, site_6,  site_15]

#Creating lists of subproblem objects corresponding to the lists of sites above
sub_problem_list = [Model(site_objects=site) for site in large_sites_list]
medium_sub_problem_list = [Model(site_objects=site) for site in medium_sites_list]
short_sub_problem_list = [Model(site_objects=site) for site in short_sites_list]

#Declaring which variables to set to one in the first node in the branch and price search tree.
#All sites with initial biomass have their deploy variable for period 0 set to be 1
short_node_init_list = [[2,0]]
medium_node_init_list = [[0,0],[4,0],[6,0],[7,0]]
long_node_init_list = [[0,0], [2,0], [4,0], [7,0], [9,0],[10,0], [11,0], [14,0]]

"""
LOGIC FOR SELECTING THE CORRECT LIST OF SITES TO SOLVE PROBLE FOR
The correct list of sites, sub-problems and initial specs are chosen based on the input given by the user.
See the configs.py file for the logic that takes the input from the user. 
"""

#Declaring variables to be used as global variables, and set them to default to the smallest problem instance
NODE_INIT_LIST = short_node_init_list
SUB_PROBLEM_LIST = short_sub_problem_list
SITE_LIST = short_sites_list

#Based on the input from the user, this control flow sets the global variables to a given instance
if configs.INSTANCE == "SMALL":
    NODE_INIT_LIST = short_node_init_list
    SUB_PROBLEM_LIST = short_sub_problem_list
    SITE_LIST = short_sites_list

elif configs.INSTANCE == "MEDIUM":
    NODE_INIT_LIST = medium_node_init_list
    SUB_PROBLEM_LIST = medium_sub_problem_list
    SITE_LIST = medium_sites_list

elif configs.INSTANCE == "LARGE":
    NODE_INIT_LIST = long_node_init_list
    SUB_PROBLEM_LIST = sub_problem_list
    SITE_LIST = large_sites_list

else:
    print("Instance set does not match any, set to default")
