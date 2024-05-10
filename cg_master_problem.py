

class CGMasterProblem:
    def __init__(self):
        pass

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
        self.lambda_var = self.model.addVars(len(self.locations_l), len(self.iterations_k), vtype=GRB.CONTINUOUS, lb=0)
        self.penalty_var = self.model.addVars(len(self.locations_l))  # TODO: remove this variable

        # Declaring the binary tracking variables
        self.deploy_bin = self.model.addVars(len(self.locations_l), len(self.periods_t), vtype=GRB.CONTINUOUS)
        self.deploy_type_bin = self.model.addVars(len(self.locations_l), len(self.smolt_types_f), len(self.periods_t),
                                                  vtype=GRB.CONTINUOUS)
        self.harvest_bin = self.model.addVars(len(self.locations_l), len(self.smolt_types_f), 61, 61,
                                              len(self.scenarios_s), vtype=GRB.CONTINUOUS)
        self.employ_bin = self.model.addVars(len(self.locations_l), len(self.smolt_types_f), 61, 61,
                                             len(self.scenarios_s), vtype=GRB.CONTINUOUS)

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