import pandas as pd
from initialization.input_data import InputData
import initialization.parameters
from model import Model

class LShapedMasterProblem(Model):
    def __init__(self, site_objects, 
                 MAB_shadow_prices_df=pd.DataFrame(), 
                 EOH_shadow_prices_df=pd.DataFrame(), 
                 input_data=InputData(), 
                 parameters=initialization.parameters, 
                 scenario_probabilities=initialization.configs.SCENARIO_PROBABILITIES):
        
        super().__init__(site_objects, MAB_shadow_prices_df, EOH_shadow_prices_df, input_data, parameters, scenario_probabilities)


    def solve():
        #TODO: Implement
        pass
    
    def declare_variables(self):
        #TODO: Implement
        pass

    def add_objective():
        #TODO: Implement
        pass

    def add_optimality_cut():
        #TODO: Implement
        pass

    def get_variable_values():
        #TODO: Implement
        pass
