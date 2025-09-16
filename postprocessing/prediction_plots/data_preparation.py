from __future__ import annotations

import sys

from utilities import convert_var_name

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import griddata

def create_data_grids(df: pd.DataFrame, fields: list[str], flow_plane: list[str]) -> dict[str,np.ndarray]:
    """
    Creating a grid of x-y-values with the corresponding values of the variable specified by field from the supplied data.
    Args:
        df: Data supplied by a DataFrame.
        fields: Names of the output variables of choice.
        flow_plane: List containing the names of the two coordinates making up the flow plane.
    Returns:
        Dictionary of the x-, y-, and all the field-values on the evenly spaced grid.
    """
    grids: dict[str,np.ndarray] = {}
    print(f"Creating uniform grid for {flow_plane[0]} and {flow_plane[1]} ...")
    # Extract data for the RBF-interpolation
    x = df[flow_plane[0]]
    y = df[flow_plane[1]]

    arr_x: np.ndarray = x.to_numpy()
    arr_y: np.ndarray = y.to_numpy()

    x_unique = np.linspace(x.min(), x.max(), 400)
    y_unique = np.linspace(y.min(), y.max(), 400)

    grid_x, grid_y = np.meshgrid(x_unique, y_unique)
    print("... done.")

    old_points: np.ndarray = np.column_stack((arr_x, arr_y))
    new_points: np.ndarray = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    for field in fields:
        print(f"Creating uniform grid for {field} ...")
        arr_field: np.ndarray[float] = df[field].to_numpy()
        grid_field_linear = griddata(points=old_points, values=arr_field,
                                     xi=new_points, method="linear"
                                     )
        # If the points on the new evenly spaced grid are out of the initial grids convex hull, grid_field_linear
        # will be NaN. Hence, the workaround with grid_field_nearest.
        grid_field_nearest = griddata(points=old_points, values=arr_field,
                                      xi=new_points, method="nearest"
                                      )

        grid_field = np.where(np.isnan(grid_field_linear), grid_field_nearest, grid_field_linear)
        grid_field = grid_field.reshape(grid_x.shape)
        grids[convert_var_name(field)] = grid_field
    print(f"... done.")

    grids[convert_var_name(flow_plane[0])] = grid_x
    grids[convert_var_name(flow_plane[1])] = grid_y

    return grids

def split_field(field: str) -> tuple[str, str]:
    split_field: list[str] = field.split(sep="(")
    label: str = split_field[0]
    unit: str = "-"
    # Field without unit doesn't have second element in split_field.
    if len(split_field) > 1:
        unit = split_field[1].split(sep=")")[0]
    return label, unit

def denormalize_data(data: pd.DataFrame, norm_stats: dict[str,dict[str,float]]) -> pd.DataFrame:
    # Columns that are not supposed to be denormalized
    exclude_columns = ["inlet_velocity"]

    denormalized_data: pd.DataFrame = data.copy()
    # Denormalizing the DataFrame
    for column in denormalized_data:
        if column in norm_stats and column not in exclude_columns:
            mean = norm_stats[column]['mean']
            std = norm_stats[column]['std']
            denormalized_data[column] = denormalized_data[column] * std + mean
    return denormalized_data

def calc_error_metrics(df_cfd: pd.DataFrame, df_pinn: pd.DataFrame, field: str):
    """
    Calculates the deviations between the predictions and actual data for the specified field.
    In addition, the probability density function of the normal distribution is determined as well.

    Parameters:
        df_cfd: DataFrame containing actual data.
        df_pinn: DataFrame predicted values.
        field: Name of the field (column) of the physical quantity of interest ('Velocity[i] (m/s)', 'Pressure (Pa)').

    Returns:
        Dictionary containing the calculated error metrics.
    """
    epsilon = 1e-9

    arr_cfd = df_cfd[field]
    arr_pinn = df_pinn[field]

    # Percentual difference
    percent_diff: pd.DataFrame = abs(((arr_cfd - arr_pinn)/(arr_cfd + epsilon))) * 100

    # Absolute difference
    diff: pd.DataFrame = arr_cfd - arr_pinn

    # Mean of the absolute deviation
    mean_deviation: float = diff.mean()

    # Mean of the percentual deviation
    mean_percent_deviation: float = percent_diff.mean()

    # Mean Squared Error (MSE)
    mse: float = (diff ** 2).mean()

    # Standard deviation
    # ddof: Delta Degrees of Freedom -> Allows for unbiased estimator (division by N-1 instead of N).
    std_deviation: float = np.std(diff, ddof=1)

    # Probability density function (PDF) of the difference between the CFD data and the predictions of the neural
    # network
    pdf_values: np.ndarray = norm.pdf(diff, loc=mean_deviation, scale=std_deviation)

    np.set_printoptions(threshold=sys.maxsize)

    results = {
        'data': arr_cfd.to_list(),
        'data_predicted': arr_pinn.to_list(),
        'diff': diff.to_list(),
        'percent_diff_min': percent_diff.min(),
        'percent_diff_max': percent_diff.max(),
        'mean_deviation': mean_deviation,
        'mean_percent_deviation': mean_percent_deviation,
        'mse': mse,
        'std_deviation': std_deviation,
        'pdf_values': pdf_values.tolist()  # PDF of the differences
    }
    return results