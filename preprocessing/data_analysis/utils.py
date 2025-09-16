from __future__ import annotations

import matplotlib.tri
from cycler import cycler

from postprocessing.general_settings import *
from postprocessing.prediction_plots.settings import markers, marker_colors

from matplotlib.axis import Axis
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import tensorflow as tf
from tkinter import filedialog

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utilities import (
    convert_var_name_rev, convert_var_for_plot_axis, convert_var_list_rev, split_collocation_and_boundary
)

def load_data(extra: str = "") -> pd.DataFrame:
    if extra != "":
        extra += " "
    model_path: str = filedialog.askopenfilename(title=f"Choose the file of the {extra}CFD-data you want to analyze.")
    return pd.read_csv(model_path, delimiter=",")

def select_data(data: pd.DataFrame, var: str) -> np.ndarray[float]:
    return data[[convert_var_name_rev(var)]].to_numpy()[:, 0]

def normalize_data(data: pd.DataFrame, var: str) -> np.ndarray[float]:
    d: np.ndarray = select_data(data, var)
    return (d - d.mean()) / d.std(ddof=1)

def to_even_grid(arr_x: np.ndarray, arr_y: np.ndarray,
                 arr_field: np.ndarray[float]) -> tuple[np.ndarray, ...]:
    grid_points: int = 400
    x_unique = np.linspace(arr_x.min(), arr_x.max(), grid_points)
    y_unique = np.linspace(arr_y.min(), arr_y.max(), grid_points)

    grid_x, grid_y = np.meshgrid(x_unique, y_unique)

    grid_field_linear = griddata(points=np.column_stack((arr_x, arr_y)), values=arr_field,
                                 xi=(grid_x, grid_y), method="linear",
                                 )
    # If the points on the new evenly spaced grid are out of the initial grids convex hull, grid_field_linear
    #       will be NaN. Hence, the workaround with grid_field_nearest.
    grid_field_nearest = griddata(points=np.column_stack((arr_x, arr_y)), values=arr_field,
                                  xi=(grid_x, grid_y), method="nearest"
                                  )

    return grid_x, grid_y, np.where(np.isnan(grid_field_linear), grid_field_nearest, grid_field_linear)

def print_stats(data: np.ndarray, field: str) -> None:
    print(f"{field}", data)
    print(f"{field}_min: {data.min()}\t {field}_max: {data.max()}")
    print(f"{field}_mean: {data.mean()}\t {field}_std: {data.std(ddof=1)}")

def plot(grid_x: np.ndarray, grid_y: np.ndarray, grid_field: np.ndarray, var: str, var_unit: str = "",
         use_log: bool = False, fig: Figure | None = None, axis: Axis | None = None,
         vmin: float | None = None, vmax: float | None = None, show_colorbar: bool = True, label_top: bool = False
         ) -> tuple[Figure, Axes, float | None, float | None]:
    if fig is None or axis is None:
        fig, axis = plt.subplots(figsize=figsize_oc)

    x_label, x_unit = convert_var_for_plot_axis(
        convert_var_name_rev("z")
    )
    x_unit, x_data = convert(x_unit, grid_x)
    axis.set_xlabel(f"{x_label} / {x_unit}")

    y_label, y_unit = convert_var_for_plot_axis(
        convert_var_name_rev("y")
    )
    y_unit, y_data = convert(y_unit, grid_y)
    axis.set_ylabel(f"{y_label} / {y_unit}")

    if var_unit == "":
        field_label, field_unit = convert_var_for_plot_axis(
            convert_var_name_rev(var)
        )
    else:
        field_label, field_unit = var, var_unit
    field_unit, field_data = convert(field_unit, grid_field)

    pcm = axis.pcolormesh(x_data, y_data, field_data,
                          cmap="jet",
                          vmin=vmin,
                          vmax=vmax,
                          rasterized=True,
                          norm=LogNorm() if use_log else None
                          )
    if show_colorbar:
        cbar = fig.colorbar(pcm, ax=axis, orientation='vertical', label=f"{field_label} / {field_unit}")

        cbar.ax.grid(False)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(direction='out')

    if label_top:
        axis.xaxis.set_label_position("top")
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", which="both", top=True, bottom=True)

    return fig, axis, grid_field.min(), grid_field.max()

def plot_split_inputs(
        boundary_inputs: np.ndarray, collocation_inputs: np.ndarray, x_str: str, y_str: str) -> Figure:
    fig, axis = plt.subplots(figsize=figsize_oc)

    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)

    x_label, x_unit = convert_var_for_plot_axis(
        convert_var_name_rev(x_str)
    )
    _, x_vals_bound = convert(x_unit, boundary_inputs[:, 0])
    x_unit, x_vals_coll = convert(x_unit, collocation_inputs[:, 0])
    axis.set_xlabel(f"{x_label} / {x_unit}")

    y_label, y_unit = convert_var_for_plot_axis(
        convert_var_name_rev(y_str)
    )
    _, y_vals_bound = convert(y_unit, boundary_inputs[:, 1])
    y_unit, y_vals_coll = convert(y_unit, collocation_inputs[:, 1])
    axis.set_ylabel(f"{y_label} / {y_unit}")

    markers_pos=["o","s"]

    axis.scatter(
        x=x_vals_bound, y=y_vals_bound,
        color=color_for_plot(marker_colors[0]), marker=markers_pos[0], s=4, label="Boundary",
        zorder=3
    )

    axis.scatter(
        x=x_vals_coll, y=y_vals_coll,
        color=color_for_plot(marker_colors[1]), marker=markers_pos[1], s=4, label="Collocation"
    )

    axis.legend(
        title="Point Types",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=2
    )

    return fig

def plot_1D_subset(arr_x: np.ndarray, arr_y: np.ndarray, arr_field: np.ndarray, pos_x: float,
                   fig: Figure | None = None, axis: Axis | None = None
                   ) -> tuple[np.ndarray, Figure]:
    pos_mask: np.ndarray = np.abs(arr_x - pos_x) < 1e-3
    x: np.ndarray = arr_y[pos_mask]
    y: np.ndarray = arr_field[pos_mask]

    sort_idx: np.ndarray = np.argsort(x)
    x_sorted: np.ndarray = x[sort_idx]
    y_sorted: np.ndarray = y[sort_idx]

    coeffs: np.ndarray = np.polyfit(x_sorted, y_sorted, 5)

    model: np.ndarray = np.polyval(coeffs, x_sorted)

    if fig is None or axis is None:
        fig, axis = plt.subplots(figsize=figsize_oc)

    x_label, x_unit = convert_var_for_plot_axis(
        convert_var_name_rev("y")
    )
    x_unit, x_vals = convert(x_unit, x_sorted)
    axis.set_ylabel(f"{x_label} / {x_unit}")

    y_label, y_unit = r"$R_\mathrm{el}$", r"$\Omega$"
    _, y_vals_true = convert(y_unit, y_sorted)
    y_unit, y_vals_model = convert(y_unit, model)
    axis.set_xlabel(f"{y_label} / {y_unit}")

    axis.axhline(y=(arr_y.max() * 10 ** 3 - 0.687388), color="black", linestyle="-")

    axis.tick_params(labelleft=False)
    axis.set_ylabel(None)
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", which="both", top=True, bottom=True)

    axis.set_ylim(arr_y.min() * 10 ** 3, arr_y.max() * 10 ** 3)

    axis.plot(
        y_vals_true, x_vals,
        color=color_for_plot("LightGrey2"), linestyle="-",
        marker="o", markerfacecolor=color_for_plot("White"), markevery=10,
        label="Data"
    )

    axis.plot(
        y_vals_model, x_vals,
        color=color_for_plot("DarkRed"), linestyle="--",
        label="Model"
    )

    axis.legend(
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.97,0.52)
    )

    return coeffs, fig
