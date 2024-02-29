from dataclasses import dataclass

@dataclass
class LShapedMasterProblemVariables():
    l: int # Not sure we need this
    y: list[list[float]] # index order f, t. same unique l to be added in all subproblems from a master problem
    deploy_bin: list[int] # index order t. same unique l to be added in all subproblems from a master problem
    deploy_type_bin: list[list[int]] # index order f, t. same unique l to be added in all subproblems from a master problem
    

