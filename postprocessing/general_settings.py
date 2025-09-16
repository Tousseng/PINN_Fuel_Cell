from postprocessing.colors import colors, color_list_for_plot, color_for_plot
from postprocessing.linestyles import linestyle_list_for_plot
from utilities import convert

import os
from typing import Final
from cycler import cycler

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.axes import Axes

MAJOR_TICK_LENGTH: Final[float] = 3.5 # Length of the major ticks
MINOR_TICK_LENGTH: Final[float] = 2 # Length of the minor ticks
MAJOR_TICK_WIDTH: Final[float] = 1 # Width of the major ticks
MINOR_TICK_WIDTH: Final[float] = 0.5 # Width of the minor ticks

SPINE_THICKNESS: Final[float] = 1 # Set the thickness of the frame

FIGURE_WIDTH_OC: Final[float] = 8.5   # in cm (OC: one column)
FIGURE_WIDTH_TC: Final[float] = 19   # in cm (TC: two columns)
FIGURE_HEIGHT_OC: Final[float] = 4.69  # in cm (OC: one column)
FIGURE_HEIGHT_OC_LARGE: Final[float] = 15 # in cm (OC: one column)
FIGURE_HEIGHT_TC: Final[float] = 4.69  # in cm (TC: two columns)

def cm_to_inch(cm_size: float) -> float:
    return cm_size / 2.54

def save_plot(fig, path: str, filename: str, dpi: int = 1000):
    # Creating the entire path
    save_path: str = os.path.join(path, filename)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)  # Saves the plot as the file specified by the extension

figsize_oc: tuple[float,float] = (cm_to_inch(FIGURE_WIDTH_OC), cm_to_inch(FIGURE_HEIGHT_OC))
figsize_oc_large: tuple[float,float] = (cm_to_inch(FIGURE_WIDTH_OC), cm_to_inch(FIGURE_HEIGHT_OC_LARGE))
figsize_tc: tuple[float,float] = (cm_to_inch(FIGURE_WIDTH_TC), cm_to_inch(FIGURE_HEIGHT_TC))

plt.rcParams.update({"font.family": "Times New Roman",
                     "text.usetex": True, # PDFLATEX NEEDS TO BE IN THE WINDOWS PATH VARIABLE
                     "font.size": 8,
                     "axes.titlesize": 10,
                     "axes.labelsize": 8,
                     "axes.linewidth": SPINE_THICKNESS,
                     "axes.grid": True,
                     "grid.linestyle": "--",
                     "grid.linewidth":  0.5,
                     "grid.alpha": 0.7,
                     "grid.color": "gray",
                     "xtick.labelsize": 8,
                     "ytick.labelsize": 8,
                     "lines.linewidth": 1,
                     "lines.markersize": 3,
                     "legend.fontsize": 8,
                     "legend.handlelength": 1.7,
                     "legend.handletextpad": 0.5,
                     "legend.borderpad": 0.2,
                     "legend.columnspacing": 0.6,
                     "legend.labelspacing": 0.3,
                     "legend.facecolor": "white",
                     "legend.edgecolor": "black",
                     "legend.fancybox": False,
                     "legend.shadow": False,
                     "legend.framealpha": 1.0,
                     "xtick.major.size": MAJOR_TICK_LENGTH,
                     "xtick.minor.size": MINOR_TICK_LENGTH,
                     "xtick.major.width": MAJOR_TICK_WIDTH,
                     "xtick.minor.width": MINOR_TICK_WIDTH,
                     "xtick.direction": "in",
                     "xtick.top": True,
                     "ytick.major.size": MAJOR_TICK_LENGTH,
                     "ytick.minor.size": MINOR_TICK_LENGTH,
                     "ytick.major.width": MAJOR_TICK_WIDTH,
                     "ytick.minor.width": MINOR_TICK_WIDTH,
                     "ytick.direction": "in",
                     "ytick.right": True
                     }
                    )