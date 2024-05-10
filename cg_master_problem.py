import gurobipy as gp
from gurobipy import GRB
import initialization.parameters as parameters
import initialization.configs as configs



class CGMasterProblem:
    def __init__(self):
        self.iterations_k = 0
        self.l_size = configs.NUM_LOCATIONS
        self.f_size = configs.NUM_SMOLT_TYPES
        self.t_size = parameters.number_periods
        self.s_size = configs.NUM_SCENARIOS
        self.columns = {} #dict, key = iterations, item = list of columns where index is l (site)

    """
    Initializing and updating model
    """
    def initialize_model(self):
        self.model = gp.Model(f"CG master problem model")
        self.declare_variables()
        self.set_objective()

    def update_model(self):
        pass

    def declare_variables(self):
        self.lambda_var = self.model.addVars(configs.NUM_LOCATIONS, self.iterations_k, vtype=GRB.CONTINUOUS, lb=0)
        
        # Declaring the tracking variables
        self.y = self.model.addVars(self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0)
        self.x = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size+1, self.s_size, vtype=GRB.CONTINUOUS, lb=0)
        self.w = self.model.addVars(self.l_size, self.f_size, self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0)

        self.deploy_bin = self.model.addVars(self.l_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.deploy_type_bin = self.model.addVars(self.l_size, self.f_size, self.t_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.employ_bin = self.model.addVars(self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.employ_bin_granular = self.model.addVars(self.t_size, self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.harvest_bin = self.model.addVars(self.t_size, self.s_size, vtype=GRB.CONTINUOUS, lb=0, ub=1)

    def solve(self):
        pass

    """
    Add objective
    """
    def set_objective(self):
        pass

    """
    Add constraints
    """

    def add_MAB_company_constraint(self):
        pass

    def add_EOH_constraint(self):
        pass

    def add_convexity_constraint(self):
        pass

    """
    Branching constraints
    """
    def add_branching_constraints(self):
        pass

    """
    Variable tracking constraints
    """
    def add_variable_tracking_constraints(self):
        pass