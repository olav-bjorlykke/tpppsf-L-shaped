from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
import initialization.configs as configs
import initialization.parameters as parameters

@dataclass
class LShapedMasterProblemVariables:
    l: int # Not sure we need this
    y: list[list[float]] # index order f, t. same unique l to be added in all subproblems from a master problem
    deploy_bin: list[int] # index order t. same unique l to be added in all subproblems from a master problem
    deploy_type_bin: list[list[int]] # index order f, t. same unique l to be added in all subproblems from a master problem

    def write_to_file(self):
        writer = pd.ExcelWriter(f"{configs.OUTPUT_DIR}master_variables_site{self.l}.xlsx")
        pd.DataFrame(self.y).to_excel(writer, index=False, sheet_name="deploy_amounts")
        pd.Series(self.deploy_bin).to_excel(writer, index=False, sheet_name="deploy_bins")
        writer.close()

@dataclass
class LShapedSubProblemDualVariables:
    rho_1: list[float] # index order t. Also needs s as scenario index
    rho_2: list[list[float]] # index order f, t. Also needs s as scenario index
    rho_3: list[float] # index order t. Also needs s as scenario index
    rho_4: float # Only needs scenario index s 
    rho_5: list[list[float]] # index order t_hat, t. Also needs s as scenario index
    rho_6: list[float] # index order t. Also needs s as scenario index
    rho_7: list[float] # index order t. Also needs s as scenario index

@dataclass
class CGDualVariablesFromMaster:
    iteration: int = field(default=0)
    u_MAB: list[list[float]] = field(default_factory=lambda: [[0.0 for s in range(configs.NUM_SCENARIOS)]for t in range(parameters.number_periods + 1)]) #t, s
    u_EOH: list[float] = field(default_factory=lambda: [0.0 for s in range(configs.NUM_SCENARIOS)])#s

    def write_to_file(self):
        writer = pd.ExcelWriter(f"{configs.OUTPUT_DIR}dual_variables_iteration{self.iteration}.xlsx")
        pd.DataFrame(self.u_MAB).to_excel(writer, index=False, sheet_name="u_MAB")
        pd.Series(self.u_EOH).to_excel(writer, index=False, sheet_name="u_EOH")
        writer.close()

@dataclass
class DeployPeriodVariables:
    y: list[list[float]] = field(default_factory= lambda: [[0.0 for t in range(parameters.number_periods)] for f in range(configs.NUM_SMOLT_TYPES)])  # Index order: f, t
    x: list[list[list[float]]] = field(default_factory=lambda: [[[0.0 for s in range(configs.NUM_SCENARIOS)] for t in range(parameters.number_periods + 1)] for f in range(configs.NUM_SMOLT_TYPES)])# Index order: f, t, s
    w: list[list[list[float]]] = field(default_factory=lambda: [[[0.0 for s in range(configs.NUM_SCENARIOS)] for t in range(parameters.number_periods)] for f in range(configs.NUM_SMOLT_TYPES)]) #Index order: f, t, s
    deploy_bin: list[float] = field(default_factory=lambda: [0.0 for t in range(parameters.number_periods)]) #Index order: t
    deploy_type_bin: list[list[float]] = field(default_factory=lambda: [[0.0 for t in range(parameters.number_periods)] for f in range(configs.NUM_SMOLT_TYPES)])#Index order: f, t
    employ_bin: list[list[float]] = field(default_factory=lambda:[[0.0 for s in range(configs.NUM_SCENARIOS)]for t in range(parameters.number_periods)]) #Index order: t, s
    employ_bin_granular: list[list[float]] = field(default_factory=lambda:[[0.0 for s in range(configs.NUM_SCENARIOS)]for t in range(parameters.number_periods)]) #Index order: t, s
    harvest_bin: list[list[float]] = field(default_factory=lambda:[[0.0 for s in range(configs.NUM_SCENARIOS)]for t in range(parameters.number_periods)]) #Index order: t, s

@dataclass
class CGColumn:
    site: int
    iteration_k: int
    production_schedules: Dict[int, DeployPeriodVariables] = field(default_factory=dict)
    def write_to_file(self):
        deploy_period_list = []
        for deploy_period, deploy_period_variables in self.production_schedules.items():
            f_list = []
            for f in range(configs.NUM_SMOLT_TYPES):
                t_list = []
                for t in range(parameters.number_periods):
                    s_list = []
                    for s in range(configs.NUM_SCENARIOS):
                        y = deploy_period_variables.y[f][t]
                        x = deploy_period_variables.x[f][t][s]
                        w = deploy_period_variables.w[f][t][s]
                        deploy_bin = deploy_period_variables.deploy_bin[t]
                        deploy_type_bin = deploy_period_variables.deploy_type_bin[f][t]
                        employ_bin = deploy_period_variables.employ_bin[t][s]
                        employ_bin_gran = deploy_period_variables.employ_bin_granular[t][s]
                        harvest_bin = deploy_period_variables.harvest_bin[t][s]
                        variable_list = [y, x, w, deploy_bin, deploy_type_bin, employ_bin, employ_bin_gran, harvest_bin]
                        s_list.append(variable_list)
                    columns = ["Y", "X", "W", "deploy_bin", "deploy_type_bin", "employ_bin", "employ_bin_gran",
                               "harvest_bin"]
                    t_list.append(
                        pd.DataFrame(s_list, columns=columns, index=[i for i in range(configs.NUM_SCENARIOS)]))
                f_list.append(pd.concat(t_list, keys=[t for t in range(parameters.number_periods)]))
            df = pd.concat(f_list, keys=[i for i in range(configs.NUM_SMOLT_TYPES)])
            df_filtered =  df #df.loc[~(df[["X", "W"]] == 0).all(axis=1)]
            deploy_period_list.append(df_filtered)
        df = pd.concat(deploy_period_list, keys=list(self.production_schedules.keys()))
        df.index.names = ["Deploy Period", "Smolt Type", "Period", "Scenario"]
        df_reordered = df.reorder_levels(["Scenario", "Smolt Type", "Deploy Period", "Period"])
        df_sorted = df_reordered.sort_index(level=["Scenario", "Smolt Type", "Deploy Period", "Period"], ascending=[True,True,True,True])
        df_sorted.to_excel(f"{configs.OUTPUT_DIR}column_variable_values_site{self.site}_iteration{self.iteration_k}.xlsx")

@dataclass
class NodeLabel:
    number: int
    parent: int
    level: int
    LP_solution: float = field(default=0)
    MIP_solution: float = field(default=0)
    up_branching_indices: list[list[int]] = field(default_factory= lambda: [[] for _ in range(configs.NUM_LOCATIONS)]) #site, index
    down_branching_indices: list[list[int]] = field(default_factory= lambda: [[] for _ in range(configs.NUM_LOCATIONS)]) #site, index


