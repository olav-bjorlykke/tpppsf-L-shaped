from dataclasses import dataclass
import pandas as pd
import initialization.configs as configs

@dataclass
class LShapedMasterProblemVariables():
    l: int # Not sure we need this
    y: list[list[float]] # index order f, t. same unique l to be added in all subproblems from a master problem
    deploy_bin: list[int] # index order t. same unique l to be added in all subproblems from a master problem
    deploy_type_bin: list[list[int]] # index order f, t. same unique l to be added in all subproblems from a master problem

    def write_to_file(self):
        writer = pd.ExcelWriter(f"{configs.OUTPUT_DIR}master_variables_site{self.l}.xlsx")
        pd.DataFrame(self.y).to_excel(writer, index=False, sheet_name="deploy_amounts")
        pd.Series(self.deploy_bin).to_excel(writer, index=False, sheet_name="deploy_bins")

    

@dataclass
class LShapedSubProblemDualVariables(): 
    rho_1: list[float] # index order t. Also needs s as scenario index
    rho_2: list[list[float]] # index order f, t. Also needs s as scenario index
    rho_3: list[float] # index order t. Also needs s as scenario index
    rho_4: float # Only needs scenario index s 
    rho_5: list[list[float]] # index order t_hat, t. Also needs s as scenario index
    rho_6: list[float] # index order t. Also needs s as scenario index
    rho_7: list[float] # index order t. Also needs s as scenario index

