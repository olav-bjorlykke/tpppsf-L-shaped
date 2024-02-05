
import gurobipy as gp
from gurobipy import GRB
import numpy as np

#Create model
model = gp.Model("vrp-model")

#Setting sets
n = 10
c = np.random.uniform(low=1, high=10, size=(n,n + 1))
q = np.random.uniform(low=0, high=100, size=(n + 1))
q[0] = 0
q[n] = 0
K = 5
Q = 200

#Declaring variables
x = model.addVars(n + 1, n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.BINARY, name="x")
y = model.addVars(n + 1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

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

def run_model(model, n, c):
    add_objective(model, c, x, n)
    add_visit_constraint(model, x, n)
    add_flow_constraint(model, x, n)
    add_route_limiter_constraint(model, x, n, K)
    add_vehicle_capacity_constraint_1(model, x, y, n, q, Q)
    add_vehicle_capacity_constraint_2(model, y, n, 1,Q)
    model.optimize()

run_model(model,n, c)

for var in model.getVars():
    if var.X != 0:
        print(f"{var.VarName} = {var.X}")
