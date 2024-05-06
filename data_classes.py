from dataclasses import dataclass
import pandas as pd

@dataclass
class LShapedMasterProblemVariables():
    l: int # Not sure we need this
    y: list[list[float]] # index order f, t. same unique l to be added in all subproblems from a master problem
    deploy_bin: list[int] # index order t. same unique l to be added in all subproblems from a master problem
    deploy_type_bin: list[list[int]] # index order f, t. same unique l to be added in all subproblems from a master problem

    def print(self):
        print(f"Location {self.l}")
        print(f"deploy amounts =\n {pd.DataFrame(self.y)}")
        print(f"deploy bin =\n {pd.Series(self.deploy_bin)}")

        pd.DataFrame(self.y).to_excel(f"master_variables_site{self.l}.xlsx", index=False, sheet_name="deploy_amounts")
        pd.Series(self.deploy_bin).to_excel(f"master_variables_site{self.l}.xlsx", index=False, sheet_name="deploy_bins")

    

@dataclass
class LShapedSubProblemDualVariables(): 
    rho_1: list[float] # index order t. Also needs s as scenario index
    rho_2: list[list[float]] # index order f, t. Also needs s as scenario index
    rho_3: list[float] # index order t. Also needs s as scenario index
    rho_4: float # Only needs scenario index s 
    rho_5: list[list[float]] # index order t_hat, t. Also needs s as scenario index
    rho_6: list[float] # index order t. Also needs s as scenario index
    rho_7: list[float] # index order t. Also needs s as scenario index

