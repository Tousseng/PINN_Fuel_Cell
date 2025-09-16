from postprocessing.general_settings import *
from log_handling import LossLogParser

from typing import Iterator

loss_comps_colors: list[str] = ["Black5", "DarkBlue5", "Green5", "Orange5", "DarkGrey5"]

model_colors: list[str] = ["DarkGrey1", "DarkGrey2", "DarkGrey3", "DarkGrey4", "DarkGrey5"]
epochs_colors: list[str] = ["DarkBlue1", "DarkBlue2", "DarkBlue3", "DarkBlue4", "DarkBlue5"]
iterations_colors: list[str] = ["Grey1", "Grey2", "Grey3", "Grey4", "Grey5"]
batch_size_colors: list[str] = ["LightBlue1", "LightBlue2", "LightBlue3", "LightBlue4", "LightBlue5"]
learning_rate_colors: list[str] = ["DarkBlueGreen1", "DarkBlueGreen2", "DarkBlueGreen3", "DarkBlueGreen4", "DarkBlueGreen5"]
layers_colors: list[str] = ["Green1", "Green2", "Green3", "Green4", "Green5"]
neurons_colors: list[str] = ["Orange1", "Orange2", "Orange3", "Orange4", "Orange5"]
best_colors: list[str] = ["Red"]

lines: list[str] = ["solid", "dashed", "dashdot1", "dotted", "dashdot2"]

markers: list[str] = ["s", "^", "D", "p", "o"]

markers_iter: Iterator[str] = iter(markers)

comp_colors: dict[str,str] = {
    "total": "Black",
    "conti": "DarkBlue4",
    "momentum": "LightBlue4",
    "momentum_bc_u": "LightBlue4",
    "bc_u": "LightBlue4",
    "momentum_bc_w": "LightBlue4",
    "bc_w": "LightBlue4",
    "momentum_bc_p": "LightBlue4",
    "bc_p": "LightBlue4",
    "spec_trans": "DarkBlueGreen3",
    "spec_trans_bc": "DarkBlueGreen3",
    "bc_Y_O2": "DarkBlueGreen3",
    "surf_pot": "Green4",
    "curr_dens": "Green3",
    "width": "Orange3",
    "data": "Grey3"
}

color_scheme: dict[str,list[str]] = {
    "model": model_colors,
    "epochs": epochs_colors,
    "iterations": iterations_colors,
    "batch_size": batch_size_colors,
    "learning_rate": learning_rate_colors,
    "layers": layers_colors,
    "neurons": neurons_colors,
    "-42": best_colors
}