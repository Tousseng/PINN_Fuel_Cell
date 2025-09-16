from utilities import convert_var_name_rev
from utilities.internal_translation import convert_var_list_rev

import os
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from tkinter import filedialog

params: list[str] = convert_var_list_rev(["rho", "mu"])
diffs: list[str] = convert_var_list_rev(["D_H2", "D_O2"])

def avg_params() -> None:
    file_name: str = os.path.basename(file_path)
    electrode: str = file_name.split("_", maxsplit=1)[0]
    param_df: pd.DataFrame = pd.read_csv(file_path)
    cols: list[str] = list(param_df.columns)

    diff: pd.DataFrame = param_df[params[1]] / param_df[params[0]]

    if electrode == "Anode" and diffs[0] not in cols:
        param_df[diffs[0]] = diff
    if electrode == "Cathode" and diffs[1] not in cols:
        param_df[diffs[1]] = diff

    params.append(diffs[0] if electrode == "Anode" else diffs[1] if electrode == "Cathode" else "")

    param_df = param_df[params].reset_index(drop=True)
    param_mean: pd.Series = param_df.mean(axis=0)
    print(param_mean.to_string(float_format="{:.3e}".format)) # Just a type checking error

def mass_flux_average(param: str) -> float:
    df: pd.DataFrame = pd.read_csv(file_path)

    p: np.ndarray = df[[convert_var_name_rev(param)]].to_numpy()
    w: np.ndarray = df[[convert_var_name_rev("w")]].to_numpy()
    x: np.ndarray = df[[convert_var_name_rev("x")]].to_numpy()
    z: np.ndarray = df[[convert_var_name_rev("z")]].to_numpy()
    pos_mask: np.ndarray = np.abs(z - z.max()) < 1e-3

    p_in: np.ndarray = p[pos_mask]
    w_in: np.ndarray = w[pos_mask]
    x_in: np.ndarray = x[pos_mask]

    sort_idx: np.ndarray = np.argsort(x_in)
    p_in_sorted: np.ndarray = p_in[sort_idx]
    w_in_sorted: np.ndarray = w_in[sort_idx]
    x_in_sorted: np.ndarray = x_in[sort_idx]

    p_w_int: float = trapezoid(p_in_sorted * w_in_sorted, x_in_sorted)
    w_int: float = trapezoid(w_in_sorted, x_in_sorted)

    return p_w_int / w_int

if __name__ == "__main__":
    file_path = filedialog.askopenfilename(title="Choose CSV files to average the parameters over.")
    df: pd.DataFrame = pd.read_csv(file_path)

    avg_params()

    #rho_avg: float = mass_flux_average("rho")
    #print(rho_avg)