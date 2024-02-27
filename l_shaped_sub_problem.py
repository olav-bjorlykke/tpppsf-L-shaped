import pandas as pd
from initialization.input_data import InputData
import initialization.configs
from model import Model

class LShapedSubProblem(Model):
    def __init__(self,
                 scenario,
                 site_objects, 
                 MAB_shadow_prices_df=pd.DataFrame(), 
                 EOH_shadow_prices_df=pd.DataFrame(), 
                 input_data=InputData(), 
                 parameters=initialization.parameters, 
                 scenario_probabilities=initialization.configs.SCENARIO_PROBABILITIES):
        
        self.scenario: scenario
        super().__init__(site_objects, MAB_shadow_prices_df, EOH_shadow_prices_df, input_data, parameters, scenario_probabilities)

    
    def solve():
        #TODO: Implement
        pass
    
    def declare_variables(self):
        #TODO: Implement with only continous variables
        pass

    def add_objective():
        #TODO: Implement
        pass

    def add_fallowing_constraints(self):
        #TODO: Implement with slack variable and fixed gamma (74)
        pass

    def add_biomass_development_constraints(self):
        #TODO: Implement with fixed y (77)
        pass

    def add_MAB_requirement_constraint(self):
        #TODO: Implement with slack variable (82)
        pass

    def get_dual_values():
        #TODO: Implement
        pass