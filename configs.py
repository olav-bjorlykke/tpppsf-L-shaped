import os
def set_instance():
    instance = int(input("Set the instance 0 = small, 1 = medium, 2 = large: "))
    return instance

def set_scenarios():
    num_scenarios = int(input("Set the number of scenarios 2,5 or 10: "))
    return num_scenarios


INSTANCES = ["SMALL", "MEDIUM", "LARGE"]
INSTANCE = INSTANCES[set_instance()]
NUM_SCENARIOS = set_scenarios()
OUTPUT_DIR = f"./output/instance_{INSTANCE}_scenario_{NUM_SCENARIOS}"


print(f"Running {NUM_SCENARIOS} scenarios and {INSTANCE} instance")

if NUM_SCENARIOS == 2:
    SCENARIOS_VARIATIONS = [0.98,1.02]
    SCENARIO_PROBABILITIES = [0.5,0.5]
elif NUM_SCENARIOS == 5:
    SCENARIOS_VARIATIONS = [0.96, 0.98, 1.0, 1.02, 1.04]
    SCENARIO_PROBABILITIES = [0.1, 0.2, 0.4, 0.2, 0.1]

elif NUM_SCENARIOS == 10:
    SCENARIOS_VARIATIONS = [0.95,0.96,0.97,0.98,0.99,1.0,1.01,1.02,1.03,1.04]
    SCENARIO_PROBABILITIES = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]


MAB_COMPANY_LIMIT = 4000 * 1000
NUM_LOCATIONS = 3

if INSTANCE == "LARGE":
    NUM_LOCATIONS = 16
    MAB_COMPANY_LIMIT = 17000 * 1000
elif INSTANCE == "MEDIUM":
    NUM_LOCATIONS = 8
    MAB_COMPANY_LIMIT = 10000 * 1000
elif INSTANCE == "SMALL":
    NUM_LOCATIONS = 3
    MAB_COMPANY_LIMIT = 4000 * 1000
else:
    print("Instance not defined")


if not os.path.exists(OUTPUT_DIR):
    # Create the directory
    os.makedirs(OUTPUT_DIR)
    print(f"Directory created: {OUTPUT_DIR}")
else:
    print(f"Directory already exists: {OUTPUT_DIR}")



