import os


class Configs:
    def __init__(self, scenarios=None, instance=None, algorithm=None, random_scenearios=False, average_values=False) -> None:
        self.ALGORITHMS_LIST = ["B&P w L-SHAPED", "B&P w GUROBI", "MONOLITHIC MODEL"]
        self.INSTANCES = ["SMALL", "MEDIUM", "LARGE", "TEST", "SINGLE_SITE"]
        self.RANDOM_SCENARIOS = random_scenearios
        self.AVERAGE_VALUES = average_values
        if instance is None:
            self.INSTANCE = self.INSTANCES[self.set_instance()]
        else:
            self.INSTANCE = instance

        if scenarios is None:
            self.NUM_SCENARIOS = self.set_scenarios()
        else:
            self.NUM_SCENARIOS = scenarios

        if algorithm is None:
            self.ALGORITHM = self.set_algorithm()
        else:
            self.ALGORITHM = algorithm

        self.OUTPUT_DIR = f"./output/instance_{self.INSTANCE}_scenario_{self.NUM_SCENARIOS}_{self.ALGORITHMS_LIST[self.ALGORITHM].strip()}/"
        self.LOG_DIR = f"{self.OUTPUT_DIR}logs/"
        self.NUM_SMOLT_TYPES = 1
        self.MAB_COMPANY_LIMIT = 5000 * 1000
        self.NUM_LOCATIONS = 1
        
        self.SCENARIOS_VARIATIONS = [0.95 + (i*0.1)/self.NUM_SCENARIOS for i in range(self.NUM_SCENARIOS)]
        self.SCENARIO_PROBABILITIES = [1/self.NUM_SCENARIOS for _ in range(self.NUM_SCENARIOS)]
        
        if self.NUM_SCENARIOS == 1:
            self.SCENARIOS_VARIATIONS = [1.0]
            self.SCENARIO_PROBABILITIES = [1.0]
        elif self.NUM_SCENARIOS == 2:
            self.SCENARIOS_VARIATIONS = [0.98,1.02]
            self.SCENARIO_PROBABILITIES = [0.5,0.5]
        elif self.NUM_SCENARIOS == 5:
            self.SCENARIOS_VARIATIONS = [0.96, 0.98, 1.0, 1.02, 1.04]
            self.SCENARIO_PROBABILITIES = [0.1, 0.2, 0.4, 0.2, 0.1]
        elif self.NUM_SCENARIOS == 10:
            self.SCENARIOS_VARIATIONS = [0.95,0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04]
            self.SCENARIO_PROBABILITIES = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    
        if self.INSTANCE == "LARGE":
            self.NUM_LOCATIONS = 16
            self.MAB_COMPANY_LIMIT = 17000 * 1000
        elif self.INSTANCE == "MEDIUM":
            self.NUM_LOCATIONS = 8
            self.MAB_COMPANY_LIMIT = 10000 * 1000
        elif self.INSTANCE == "SMALL":
            self.NUM_LOCATIONS = 3
            self.MAB_COMPANY_LIMIT = 6000 * 1000
        elif self.INSTANCE == "TEST":
            self.NUM_LOCATIONS = 2
            self.MAB_COMPANY_LIMIT = 5000 * 1000
        else:
            self.NUM_LOCATIONS = 1
            self.MAB_COMPANY_LIMIT = 5000 * 1000


        if not os.path.exists(self.OUTPUT_DIR):
            # Create the directory
            os.makedirs(self.OUTPUT_DIR)
            print(f"Directory created: {self.OUTPUT_DIR}")
        else:
            print(f"Directory already exists: {self.OUTPUT_DIR}")

        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        else:
            print(f"Directory already exists: {self.LOG_DIR}")

        print(f"Running {self.NUM_SCENARIOS} scenarios and {self.INSTANCE} instance with algorithm {self.ALGORITHMS_LIST[self.ALGORITHM]}.")


    def set_instance(self):
        instance = int(input("Set the instance 0 = small, 1 = medium, 2 = large, 3 = test, 4 = single site: "))
        return instance

    def set_scenarios(self):
        num_scenarios = int(input("Set the number of scenarios 1, 2, 5 or 10: "))
        return num_scenarios

    def set_algorithm(self):
        algorithm = int(input(f"SET ALGORITHM - 0 = {self.ALGORITHMS_LIST[0]}, 1 = {self.ALGORITHMS_LIST[1]}, 2 = {self.ALGORITHMS_LIST[2]}: "))
        return algorithm


    



