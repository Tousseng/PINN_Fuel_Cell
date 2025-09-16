from postprocessing.prediction_plots.settings import *
from utilities import convert_surf_avg_name
from postprocessing.prediction_plots.cfd_params import transform_react_rate_to_curr_dens

def create_surf_avgs_plot(axis: Axes, channel_widths: np.ndarray, avg_infos: dict[str,np.ndarray],
                          field: str) -> None:

    x_info: str = convert_var_name_rev("b")
    xlabel, xunit = convert_var_for_plot_axis(x_info)
    xunit, channel_widths = convert(xunit, np.array(channel_widths))

    ylabel, yunit = convert_var_for_plot_axis(
        field if field != convert_var_name_rev("j") else "Current Density (A/cm^2)"
    )

    # y-axis: Predicted value
    axis.set_xlabel(f"{xlabel} / {xunit}")
    axis.set_ylabel(f"{ylabel} / {yunit}")
    axis.tick_params(axis='y', labelcolor=color_for_plot('Black'))

    axis.set_prop_cycle(cycler(color=color_list_for_plot(surf_avgs_colors))
                       + cycler(linestyle=linestyle_list_for_plot(surf_avgs_lines))
                       + cycler(marker=surf_avgs_markers)
                       )

    for src, avg_data in avg_infos.items():
        axis.plot(
            channel_widths, avg_data, markerfacecolor=color_for_plot("White"), label=f"{src}"
        )

    # Set font for tick labels (numerical values on the axes)
    plt.xticks()  # Font and size for X-axis
    plt.yticks()  # Font and size for Y-axis

    axis.legend(
        loc="best",
        frameon=True
    )

def create_mult_surf_avgs_plot(axis: Axes, channel_widths: np.ndarray, avg_infos: dict[str,np.ndarray],
                          variation_data: dict[str,dict[str,np.ndarray]], field: str) -> None:

    x_info: str = convert_var_name_rev("b")
    xlabel, xunit = convert_var_for_plot_axis(x_info)
    xunit, channel_widths = convert(xunit, np.array(channel_widths))

    ylabel, yunit = convert_var_for_plot_axis(
        field if field != convert_var_name_rev("j") else "Current Density (A/cm^2)"
    )

    # y-axis: Predicted value
    axis.set_xlabel(f"{xlabel} / {xunit}")
    axis.set_ylabel(f"{ylabel} / {yunit}")
    axis.tick_params(axis='y', labelcolor=color_for_plot('Black'))

    for src, avg_data in avg_infos.items():
        axis.plot(
            channel_widths, avg_data,
            markerfacecolor=color_for_plot("White"),
            color=color_for_plot("DarkBlueGreen3" if "PINN" in src else "Orange3" if "ANN" in src else "Black"),
            marker=("v" if "best" in src else "^" if "worst" in src else "o"),
            linestyle=("--" if "worst" in src else "-." if "best" in src else "-"),
            label=f"{convert_surf_avg_name(src)}")

    for src, min_max_data in variation_data.items():
        axis.fill_between(
            x=channel_widths, y1=min_max_data["min"], y2=min_max_data["max"], alpha=0.5,
            color=color_for_plot("DarkBlueGreen3" if src == "PINN" else "Orange3")
        )

    # Set font for tick labels (numerical values on the axes)
    plt.xticks()  # Font and size for X-axis
    plt.yticks()  # Font and size for Y-axis

    axis.legend(
        loc="best",
        frameon=True
    )

def transform_data(avgs_data: list[float], var: str) -> list[float]:
    if var != "j":
        return avgs_data

    # Only transform Reaction Rate to Current Density
    trans_react_data: list[float] = []
    for idx in range(len(avgs_data)):
        trans_react_data.append(transform_react_rate_to_curr_dens(avgs_data[idx]))
    return trans_react_data

def get_data(model_path: str, field: str) -> dict[str,np.ndarray]:
    avg_log_parser: AvgLogParser = AvgLogParser(model_path)
    pinn_log_parser: PINNLogParser = PINNLogParser(model_path)
    nn_name: str = "PINN" if pinn_log_parser.is_pinn() else "ANN"
    cfd_avgs: list[float] = transform_data(avg_log_parser.get_cfd_avgs(field), field)
    nn_avgs: list[float] = transform_data(avg_log_parser.get_nn_avgs(field, nn_name), field)
    return {"CFD": np.array(cfd_avgs), f"{nn_name}": np.array(nn_avgs)}

def get_channel_widths(model_path: str, field: str) -> np.ndarray:
    avg_log_parser: AvgLogParser = AvgLogParser(model_path)
    return np.array(avg_log_parser.get_channel_widths(field))

def plot_surf_avgs(var_list: list[str]) -> None:
    model_path_1: str = filedialog.askdirectory(title="Chosse a model to create plots for.")
    model_path_2: str = filedialog.askdirectory(title="Chosse a model to create plots for.")
    for var in var_list:
        field: str = convert_var_name_rev(var)
        fig, axis = plt.subplots(figsize=figsize_oc, constrained_layout=True)
        avg_infos_pinn: dict[str,np.ndarray] = get_data(model_path_1, var)
        avg_infos_ann: dict[str, np.ndarray] = get_data(model_path_2, var)
        avg_infos: dict[str,np.ndarray] = avg_infos_pinn | avg_infos_ann
        create_surf_avgs_plot(axis=axis, channel_widths=get_channel_widths(model_path_1, var),
                              avg_infos=avg_infos, field=field
                              )
        save_plot(fig=fig, path=model_path_1,
                  filename=f"Surface_average-{convert_var_for_fig_name(field)}.pdf")
        plt.close()

def plot_mult_surf_avgs(var_list: list[str]) -> None:
    surf_avgs_path: str = filedialog.askdirectory(title="Choose the directory where the surface averages reside.")
    for var in var_list:
        avg_infos: dict[str,np.ndarray] = {}
        var_data: dict[str,dict[str,np.ndarray]] = {}
        avg_infos_pinn: list[np.ndarray] = []
        avg_infos_ann: list[np.ndarray] = []
        field: str = convert_var_name_rev(var)
        last_file: str = ""
        fig, axis = plt.subplots(figsize=figsize_oc, constrained_layout=True)
        for surf_avg_file in os.listdir(surf_avgs_path):
            # Do not open the pdf-files themselves or the README.txt
            if surf_avg_file.endswith("pdf") or surf_avg_file.endswith("txt"):
                continue

            surf_avg_path: str = os.path.join(surf_avgs_path, surf_avg_file)
            curr_avg_infos: dict[str,np.ndarray] = get_data(surf_avg_path, var)
            avg_infos["CFD"] = curr_avg_infos["CFD"]
            if "PINN" in list(curr_avg_infos.keys()):
                avg_infos_pinn.append(curr_avg_infos["PINN"])
            elif "ANN" in list(curr_avg_infos.keys()):
                avg_infos_ann.append(curr_avg_infos["ANN"])

            last_file = surf_avg_file

        avg_infos_pinn_arr: np.ndarray = np.array(avg_infos_pinn)
        avg_infos_ann_arr: np.ndarray = np.array(avg_infos_ann)

        var_data["PINN"] = {
            "min": np.min(avg_infos_pinn_arr, axis=0),
            "max": np.max(avg_infos_pinn_arr, axis=0)
        }
        var_data["ANN"] = {
            "min": np.min(avg_infos_ann_arr, axis=0),
            "max": np.max(avg_infos_ann_arr, axis=0)
        }

        diff_cfd_pinn: np.ndarray = np.abs(avg_infos_pinn_arr - avg_infos["CFD"])
        diff_cfd_ann: np.ndarray = np.abs(avg_infos_ann_arr - avg_infos["CFD"])

        sum_diff_cfd_pinn: np.ndarray = np.sum(diff_cfd_pinn, axis=1)
        sum_diff_cfd_ann: np.ndarray = np.sum(diff_cfd_ann, axis=1)

        pinn_idx_best: int = np.argmin(sum_diff_cfd_pinn)
        pinn_idx_worst: int = np.argmax(sum_diff_cfd_pinn)

        ann_idx_best: int = np.argmin(sum_diff_cfd_ann)
        ann_idx_worst: int = np.argmax(sum_diff_cfd_ann)

        avg_infos["PINN_best"] = avg_infos_pinn_arr[pinn_idx_best]
        avg_infos["PINN_worst"] = avg_infos_pinn_arr[pinn_idx_worst]

        avg_infos["ANN_best"] = avg_infos_ann_arr[ann_idx_best]
        avg_infos["ANN_worst"] = avg_infos_ann_arr[ann_idx_worst]

        create_mult_surf_avgs_plot(
            axis=axis, channel_widths=get_channel_widths(os.path.join(surf_avgs_path, last_file) , var),
            avg_infos=avg_infos, field=field, variation_data=var_data
        )
        save_plot(
            fig=fig, path=surf_avgs_path, filename=f"Surface_averages-{convert_var_for_fig_name(field)}.pdf"
        )
        plt.close()

if __name__ == "__main__":
    #plot_surf_avgs(["j", "eta"])
    plot_mult_surf_avgs(["j", "eta"])