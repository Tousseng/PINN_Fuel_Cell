from __future__ import annotations

import os.path
import sys

from matplotlib.collections import QuadMesh

from postprocessing.prediction_plots.settings import *

def create_colormesh(info_dict: dict[str,dict[str, Axes|str]], x_info: str, y_info: str,
                     field: str, cmap='jet') -> tuple[dict[str,tuple[str,str]], dict[str,QuadMesh]]:
    colormeshes: dict[str,QuadMesh] = {}

    x_label, x_unit = convert_var_for_plot_axis(x_info)
    x_unit, x_data = convert(x_unit, load_arr(info_dict["CFD"]["file_path"], x_info, "CFD"))

    y_label, y_unit = convert_var_for_plot_axis(y_info)
    y_unit, y_data = convert(y_unit, load_arr(info_dict["CFD"]["file_path"], y_info, "CFD"))

    grids: dict[str,np.ndarray] = {"x": x_data,
                                   "y": y_data
                                   }

    label_infos: dict[str,tuple[str,str]] = {"x": (x_label, x_unit),
                                             "y": (y_label, y_unit)
                                             }

    min_val: float = sys.float_info.max
    max_val: float = -sys.float_info.max

    for src, val in info_dict.items():
        field_label, field_unit = convert_var_for_plot_axis(field)
        field_unit, field_data = convert(field_unit, load_arr(val["file_path"], field, src))
        grids[src] = field_data
        label_infos[field] = (field_label, field_unit)
        curr_min: float = np.min(grids[src])
        curr_max: float = np.max(grids[src])
        if curr_min < min_val:
            min_val = curr_min
        if curr_max > max_val:
            max_val = curr_max

    for src, val in info_dict.items():
        # Creation of the pcolormesh plot
        colormeshes[src] = val["axis"].pcolormesh(grids["x"], grids["y"], grids[src],
                                                  cmap=cmap,
                                                  vmin=min_val,
                                                  vmax=max_val,
                                                  rasterized=True
                                                  )

    return label_infos, colormeshes

def create_cfd_pinn_comparison(file_path: str, is_pinn: bool, fields: list[str], x_info: str, y_info: str,
                               save_path: str) -> None:
    nn_name: str = "PINN" if is_pinn else "ANN"

    for field in fields:
        fig1 = plt.figure(figsize=figsize_oc)

        gs = gridspec.GridSpec(1,3, width_ratios=[10,10,1], figure=fig1)

        axs1 = fig1.add_subplot(gs[0])
        axs2 = fig1.add_subplot(gs[1])
        caxs = fig1.add_subplot(gs[2])

        label_infos, pcms = create_colormesh(info_dict={"CFD": {"file_path": file_path, "axis": axs1},
                                                        nn_name: {"file_path": file_path, "axis": axs2}
                                                        },
                                             x_info=x_info,
                                             y_info=y_info,
                                             field=field
                                             )

        x_label, x_unit = label_infos["x"]
        plt.tight_layout(rect=(0.05, 0.03, 0.92, 0.95))
        fig1.text(x=0.5, y=0.02, s=f"{x_label} / {x_unit}", ha="center")

        y_label, y_unit = label_infos["y"]
        axs1.set_ylabel(f"{y_label} / {y_unit}")

        axs1.text(0.2, 0.92, "CFD", transform=axs1.transAxes, wrap=True,
                  verticalalignment="center", horizontalalignment="center", zorder=3)

        axs2.text(0.2, 0.92, nn_name, transform=axs2.transAxes, wrap=True,
                  verticalalignment="center", horizontalalignment="center", zorder=3)
        axs2.set_yticklabels([])

        # Adding a color bar
        colbar_label, colbar_unit = label_infos[field]
        cbar = fig1.colorbar(pcms["CFD"], cax=caxs, orientation='vertical', pad=1.0,
                      label=f"{colbar_label} / {colbar_unit}")

        cbar.ax.grid(False)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(direction='out')

        plt.subplots_adjust(wspace=0.2)

        save_plot(fig=fig1, path=save_path, filename=f"Colorplot-{convert_var_for_fig_name(field)}.pdf")

        plt.close()

def load_arr(file_path: str, field: str, src: str) -> np.ndarray:
    return np.load(os.path.join(file_path, f"{convert_var_name(field)}_{src}.npy"))


def plot_cfd_pinn_comparison(model_dir: str, fig_dir_name: str, plot_extension: str = "") -> None:
    color_log_parser: ColorLogParser = ColorLogParser(model_dir, plot_extension)
    pinn_log_parser: PINNLogParser = PINNLogParser(model_dir)
    # Interim models do not have the logs available
    if not os.path.exists(pinn_log_parser.get_log_path()):
        # Move to directories upward
        pinn_dir = os.path.abspath(os.path.join(model_dir, "..", ".."))
        pinn_log_parser = PINNLogParser(pinn_dir)

    is_pinn: bool = pinn_log_parser.is_pinn()

    create_cfd_pinn_comparison(file_path=color_log_parser.get_log_path(),
                               fields=color_log_parser.get_fields(),
                               is_pinn=is_pinn,
                               x_info=color_log_parser.get_flow_plane()[0],
                               y_info=color_log_parser.get_flow_plane()[1],
                               save_path=os.path.join(model_dir, fig_dir_name)
                               )

if __name__ == "__main__":
    model_dir = filedialog.askdirectory(title="Choose the directory of the model you want plots for.")
    fig_dir_name: str = "figures"
    plot_extension: str = ""
    if not os.path.exists(os.path.join(model_dir, fig_dir_name)):
        fig_dir_name = os.path.basename(filedialog.askdirectory(title="Choose the directory of the figures."))
        plot_extension = fig_dir_name.split("_", maxsplit=1)[1]
    plot_cfd_pinn_comparison(model_dir=model_dir,
                             fig_dir_name=fig_dir_name,
                             plot_extension=plot_extension
                             )