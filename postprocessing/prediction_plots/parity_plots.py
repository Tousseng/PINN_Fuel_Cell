import numpy as np

from postprocessing.prediction_plots.settings import *

def create_error_scatter_plot(axis: Axes, cfd_data: np.ndarray, nn_data_dict: dict[str, np.ndarray], field: str):

    label, unit = convert_var_for_plot_axis(field)

    for nn_name, nn_data in nn_data_dict.items():
        _, nn_data_dict[nn_name] = convert(unit, nn_data)

    unit, cfd_data = convert(unit, cfd_data)

    # y-axis: Predicted value
    axis.set_xlabel(f'CFD {label} / {unit}')
    axis.set_ylabel(f'Predicted {label} / {unit}')
    axis.tick_params(axis='y', labelcolor=color_for_plot('Black'))

    for idx, nn_name in enumerate(nn_data_dict.keys()):
        # Plot of the Predictions
        axis.scatter(
            x=cfd_data, y=nn_data_dict[nn_name], s=8, #s wasn't set before
            zorder=3, color=color_list_for_plot(marker_colors)[idx],
            marker=markers[idx], label=nn_name
        )

    # Plot of the mean velocity
    selected_data_mean: float = cfd_data.mean()
    axis.scatter(
        x=selected_data_mean, y=selected_data_mean, color=color_for_plot("Black"), marker='o',
        s=15,#50,
        zorder=3, label=f"Mean"
        )

    axis.plot(
        cfd_data, cfd_data, color=color_for_plot("Black"),
        label=f"CFD"
    )

    # Set font for tick labels (numerical values on the axes)
    plt.xticks()  # Font and size for X-axis
    plt.yticks()  # Font and size for Y-axis

    all_data: list[np.ndarray] = [cfd_data] + [nn_data for nn_data in nn_data_dict.values()]

    axis.set_xlim(cfd_data.min(), cfd_data.max())
    axis.set_ylim(np.min(all_data), np.max(all_data))


def plot_error_scatter(model_dir: str, fields: list[str], fig_dir_name: str, plot_extension: str = ""):
    for field in fields:
        par_log_parser: HistLogParser = HistLogParser(model_dir, plot_extension)
        pinn_log_parser: PINNLogParser = PINNLogParser(model_dir)
        # Interim models do not have the logs available
        if not os.path.exists(pinn_log_parser.get_log_path()):
            # Move to directories upward
            pinn_dir = os.path.abspath(os.path.join(model_dir, "..", ".."))
            pinn_log_parser = PINNLogParser(pinn_dir)

        is_pinn: bool = pinn_log_parser.is_pinn()

        fig, axis = plt.subplots(figsize=figsize_oc, constrained_layout=True)

        cfd_data: np.ndarray = np.array(par_log_parser.get_cfd_arr(convert_var_name(field)))
        pinn_data: np.ndarray = np.array(par_log_parser.get_pinn_arr(convert_var_name(field)))

        every: int = 10

        create_error_scatter_plot(axis=axis,
                                  cfd_data=cfd_data[::every],
                                  nn_data_dict={"PINN" if is_pinn else "ANN": pinn_data[::every]},
                                  field=field
                                  )
        axis.legend(
            loc="upper left",
            frameon=True
        )
        save_plot(fig=fig, path=os.path.join(model_dir, fig_dir_name),
                  filename=f"Parityplot-{convert_var_for_fig_name(field)}.pdf")
        plt.close()

if __name__ == "__main__":
    model_dir: str = filedialog.askdirectory(title="Chosse a model to create plots for.")
    fig_dir_name: str = "figures"
    plot_extension: str = ""
    if not os.path.exists(os.path.join(model_dir, fig_dir_name)):
        fig_dir_name = os.path.basename(filedialog.askdirectory(title="Choose the directory of the figures."))
        plot_extension = fig_dir_name.split("_", maxsplit=1)[1]
    data_log_parser: DataLogParser = DataLogParser(model_dir)
    plot_error_scatter(model_dir=model_dir,
                       fields=data_log_parser.get_outputs(),
                       fig_dir_name=fig_dir_name,
                       plot_extension=plot_extension
                       )