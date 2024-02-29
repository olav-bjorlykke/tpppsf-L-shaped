from dataclasses import dataclass

@dataclass
class LShapedMasterProblemVariables():
    l: int # Not sure we need this
    y: list[list[float]] # index order f, t. same unique l to be added in all subproblems from a master problem
    deploy_bin: list[int] # index order t. same unique l to be added in all subproblems from a master problem
    deploy_type_bin: list[list[int]] # index order f, t. same unique l to be added in all subproblems from a master problem
    

@dataclass
class LShapedSubProblemDualVariables(): 
    rho_1: list[float] # index order t. Also needs s as scenario index
    rho_2: list[list[float]] # index order f, t. Also needs s as scenario index
    rho_3: list[float] # index order t. Also needs s as scenario index
    rho_4: float # Only needs scenario index s 
    rho_5: list[list[float]] # index order t_hat, t. Also needs s as scenario index
    rho_6: list[float] # index order t. Also needs s as scenario index
    rho_7: list[float] # index order t. Also needs s as scenario index