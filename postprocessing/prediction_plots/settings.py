from postprocessing.general_settings import *
from utilities import (convert_var_for_plot_axis, convert_var_for_plot_title, convert_var_for_fig_name,
                       convert_var_name, convert_var_name_rev)
from log_handling import DataLogParser, ColorLogParser, HistLogParser, AvgLogParser, PINNLogParser, LogParser

from utilities import convert

from tkinter import filedialog

import numpy as np
import matplotlib.gridspec as gridspec

markers: list[str] = ["x", "+"]
marker_colors: list[str] = ["DarkBlue", "Orange"]

surf_avgs_markers: list[str] = ["o", "s", "^"]
surf_avgs_lines: list[str] = ["solid", "dashed", "dotted"]
surf_avgs_colors: list[str] = ["Black3", "DarkBlueGreen3", "Orange3"]