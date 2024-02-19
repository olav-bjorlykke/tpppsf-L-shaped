import gurobipy as gp
from gurobipy import GRB
import numpy as np
from multiprocessing import Process
import logging
import time


#Setting objective
def add_objective(model, c, x, n):
    model.setObjective(
        gp.quicksum(
            x[i, j] * c[i][j] for i in range(n) for j in range(1, n + 1)
        ), GRB.MINIMIZE
    )

def add_visit_constraint(model, x, n):
    model.addConstrs(
        gp.quicksum(x[i, j] for j in range(1, n + 1) if j != i) == 1
        for i in range(1, n)
    )

def add_flow_constraint(model, x, n):
    model.addConstrs(
        gp.quicksum(x[i, h] for i in range(n) if i != h) - gp.quicksum(x[h, j] for j in range(1, n + 1) if j != h) == 0
        for h in range(1, n)
    )

def add_route_limiter_constraint(model, x, n, K):
    model.addConstr(
        gp.quicksum(x[0,j] for j in range(1,n)) <= K,
        name="Route limiter"
    )

def add_vehicle_capacity_constraint_1(model, x, y, n, q, Q):
    model.addConstrs(
        y[j] >= y[i] + q[j] * x[i,j] - Q*(1 - x[i,j])
        for i in range(n+1)
        for j in range(n+1)
    )


def add_vehicle_capacity_constraint_2(model, y, n, d, Q):
    model.addConstrs(
        y[i] <= Q for i in range(n+1)
    )

def print_solution(model):
    for var in model.getVars():
        if var.X != 0:
            print(f"{var.VarName} = {var.X}")

def write_solving_time_to_file(description, solving_time,):
    with open('solving_time_to_file.txt', 'a') as file:
        file.write(description)
        file.write(solving_time)
        file.write('\n')

def run_model(c, q, n):
    model = gp.Model("vrp-model")
    model.setParam("OutputFlag", 0)
    # Setting sets
    q[0] = 0
    q[n] = 0
    K = 5
    Q = 500

    # Declaring variables
    x = model.addVars(n + 1, n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, name="x")
    y = model.addVars(n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

    add_objective(model, c, x, n)
    add_visit_constraint(model, x, n)
    add_flow_constraint(model, x, n)
    add_route_limiter_constraint(model, x, n, K)
    add_vehicle_capacity_constraint_1(model, x, y, n, q, Q)
    add_vehicle_capacity_constraint_2(model, y, n, 1,Q)

    model.optimize()

    #print_solution(model)

def run_process_in_parallel(process, args):
    processes = [Process(target=process, args=args)]
    start = time.perf_counter()
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    end = time.perf_counter()

    #Returning the time spent running all processes
    return end - start


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    num_problems = 5
    num_runs = 10

    #Setting number of nodes
    n = 18
    #Creating 5 arrays of arc-costs and demands, for 5 iterations of the problem
    costs_c = [np.random.uniform(low=1, high=10, size=(n, n + 1)) for i in range(num_problems)]
    demands_q = [np.random.uniform(low=0, high=100, size=(n + 1)) for i in range(num_problems)]

    # Start each process
    runtimes_paralell = []
    runtimes_series = []

    for i in range(num_runs):
        logging.info(f"Running itereation {i} of paralell computing")
        processes = [Process(target=run_model, args=(costs_c[j], demands_q[j], n)) for j in range(len(costs_c))]
        start_paralell = time.perf_counter()
        for p in processes:
            p.start()
        # Wait for all processes to finish
        for p in processes:
            p.join()
        end_paralell = time.perf_counter()
        runtimes_paralell.append(end_paralell - start_paralell)

    for i in range(num_runs):
        logging.info(f"Running itereation {i} of serial computing")
        start_series = time.perf_counter()
        for j in range(num_problems):
            run_model(costs_c[j], demands_q[j], n)
        end_series = time.perf_counter()
        runtimes_series.append(end_series - start_series)

    print("All processes have finished.")
    print("paralell", runtimes_paralell)
    print("series", runtimes_series)
    write_solving_time_to_file("PARALELL:", runtimes_paralell)
    write_solving_time_to_file("SERIES:", runtimes_series)



