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
    pass

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

def run_model(iter):
    model = gp.Model("vrp-model")
    model.setParam("OutputFlag", 0)
    # Setting sets
    n = 15
    c = np.random.uniform(low=1, high=10, size=(n, n + 1))
    q = np.random.uniform(low=0, high=100, size=(n + 1))
    q[0] = 0
    q[n] = 0
    K = 5
    Q = 450

    # Declaring variables
    x = model.addVars(n + 1, n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, name="x")
    y = model.addVars(n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

    add_objective(model, c, x, n)
    add_visit_constraint(model, x, n)
    add_flow_constraint(model, x, n)
    add_route_limiter_constraint(model, x, n, K)
    add_vehicle_capacity_constraint_1(model, x, y, n, q, Q)
    add_vehicle_capacity_constraint_2(model, y, n, 1,Q)

    logging.info(f"Optimizing VRP Model{iter}")
    model.optimize()
    logging.info(f"Finished solving VRP Model{iter}")

    #print_solution(model)
    return iter

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)



    # Start each process
    for i in range(5):
        processes = [Process(target=run_model, args=(i,)) for i in range(6)]
        start = time.perf_counter()
        for p in processes:
            p.start()
        # Wait for all processes to finish
        for p in processes:
            p.join()
        end = time.perf_counter()
        write_solving_time_to_file(f"paralell {i}:", str(end-start))

        start = time.perf_counter()
        for i in range(6):
            run_model(i)
        end = time.perf_counter()
        write_solving_time_to_file(f"series {i}:", str(end - start))


    print("All processes have finished.")


