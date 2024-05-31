
import pandas as pd

'''
DECLARING GLOBAL PARAMETERS
'''
smolt_deployment_upper_bound = 80000 #Upper bound for biomass of smolt deployed in kilo
smolt_deployment_lower_bound = 10000 #Lower bound of smolt deployed
max_harvest = 10000 * 1000 #Max biomass that can be harvested in any period in tons
min_harvest = 1000 * 1000 #Minimum amount of biomass that can be harvested if biomass is harvested in tons
max_harvest_company = 6000 * 1000 #Max biomass that can be havested across the company in tons, currently unlimited
expected_production_loss = 0.002 #Expected loss per period
smolt_type_df = pd.DataFrame( #All possible smolt type and weights, with corresponding number of smolt per kilo
    data=[[100,10],[150,6.66],[250,4]],
    columns=["weight","num-smolt-kilo"]
)
smolt_weights = [100,250]
min_fallowing_periods = 2
max_fallowing_periods = 36
max_periods_deployed = 24
number_periods = 60
bigM = 100000000
weight_req_for_harvest = 3000.0
MAB_util_end = 0.3
penalty_parameter_L_sub = 1000 #This should not be very hidh -> It will lead to numeric instability
valid_ineqaulity_lshaped_master_bigM = 60 #This must be higher than the max possible release periods -> Never more than 60
EOH_ratio_requirement = 0.8
random_seed = 10

