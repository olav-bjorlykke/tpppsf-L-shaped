from dataclasses import dataclass
import pandas as pd
import initialization.configs as configs
import initialization.parameters as parameters

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

@dataclass
class CGDualVariablesFromMaster():
    iteration: int
    u_MAB: list[list[float]] #t, s
    u_EOH: list[float] #s

    def write_to_file(self):
        writer = pd.ExcelWriter(f"{configs.OUTPUT_DIR}dual_variables_iteration{self.iteration}.xlsx")
        pd.DataFrame(self.u_MAB).to_excel(writer, index=False, sheet_name="u_MAB")
        pd.Series(self.u_EOH).to_excel(writer, index=False, sheet_name="u_EOH")



@dataclass
class CGColumnFromSubProblem():
    site: int
    iteration: int
    y: list[list[float]]  # Index order: f, t
    x: list[list[list[list[float]]]]  # Index order: f, t_hat, t, s
    w: list[list[list[list[float]]]] #Index order: f, t_hat, t, s
    deploy_bin: list[float] #Index order: t
    deploy_type_bin: list[list[float]] #Index order: f, t
    employ_bin: list[list[float]] #Index order: t, s
    employ_bin_granular: list[list[list[float]]] #Index order: f,t,s
    harvest_bin: list[list[float]] #Index order: t, s

    def write_to_file(self):
        f_list = []
        for f in range(configs.NUM_SMOLT_TYPES):
            t_hat_list = []
            for t_hat in range(parameters.number_periods):
                t_list = []
                for t in range(parameters.number_periods):
                    s_list = []
                    for s in range(configs.NUM_SCENARIOS):
                        y = self.y[f][t]
                        x = self.x[f][t_hat][t][s]
                        w = self.w[f][t_hat][t][s]
                        deploy_bin = self.deploy_bin[t]
                        deploy_type_bin = self.deploy_type_bin[f][t]
                        employ_bin = self.employ_bin[t][s]
                        employ_bin_gran = self.employ_bin_granular[t_hat][t][s]
                        harvest_bin = self.harvest_bin[t][s]
                        variable_list = [y, x, w, deploy_bin, deploy_type_bin, employ_bin, employ_bin_gran, harvest_bin]
                        s_list.append(variable_list)
                    columns = ["Y","X", "W", "deploy_bin", "deploy_type_bin", "employ_bin", "employ_bin_gran", "harvest_bin"]
                    t_list.append(pd.DataFrame(s_list, columns=columns, index=[i for i in range(configs.NUM_SCENARIOS)]))
                t_hat_list.append(pd.concat(t_list, keys=[i for i in range(parameters.number_periods)]))
            f_list.append(pd.concat(t_hat_list, keys=[i for i in range(parameters.number_periods)]))
        df = pd.concat(f_list, keys=[i for i in range(configs.NUM_SMOLT_TYPES)])
        df_filtered = df.loc[~(df[["X", "W"]] == 0).all(axis=1)]
        df_filtered.to_excel(f"{configs.OUTPUT_DIR}column_variable_values_site{self.site}_iteration{self.iteration}.xlsx")



