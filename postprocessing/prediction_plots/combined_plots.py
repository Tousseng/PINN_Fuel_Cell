from postprocessing.prediction_plots.settings import *
from postprocessing.prediction_plots.color_plots import create_colormesh
from postprocessing.prediction_plots.parity_plots import create_error_scatter_plot

def plot_combination(file_path_pinn: str, file_path_ann: str, x_info: str, y_info: str, field: str, save_path: str) -> None:
    fig = plt.figure(figsize=figsize_tc)

    gs = gridspec.GridSpec(2,5, width_ratios=[5,5,5,1,5], height_ratios=[1,15], hspace=0.1,
                           figure=fig
                           )

    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,4])
    cax = fig.add_subplot(gs[0,:3])

    label_infos, pcms = create_colormesh(info_dict={"CFD":  {"file_path": file_path_pinn, "axis": ax1},
                                                    "PINN": {"file_path": file_path_pinn, "axis": ax2},
                                                    "ANN":  {"file_path": file_path_ann,  "axis": ax3}
                                                    },
                                         x_info=x_info,
                                         y_info=y_info,
                                         field=field
                                         )
    x_label, x_unit = label_infos["x"]
    plt.tight_layout(rect=(0.05, 0.03, 0.92, 0.95))

    ax2.set_xlabel(f"{x_label} / {x_unit}")

    y_label, y_unit = label_infos["y"]
    ax1.set_ylabel(f"{y_label} / {y_unit}")

    ax1.text(0.2, 0.92, "CFD", transform=ax1.transAxes, wrap=True,
             verticalalignment="center", horizontalalignment="center", zorder=3)

    ax2.text(0.2, 0.92, "PINN", transform=ax2.transAxes, wrap=True,
             verticalalignment="center", horizontalalignment="center", zorder=3)
    ax2.set_yticklabels([])

    ax3.text(0.2, 0.92, "ANN", transform=ax3.transAxes, wrap=True,
             verticalalignment="center", horizontalalignment="center", zorder=3)
    ax3.set_yticklabels([])

    # Adding a color bar
    colbar_label, colbar_unit = label_infos[field]
    cbar = fig.colorbar(pcms["CFD"], cax=cax, orientation='horizontal',
                         label=f"{colbar_label} / {colbar_unit}")

    cbar.ax.grid(False)

    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(direction='out')

    par_log_parser_pinn: HistLogParser = HistLogParser(model_dir_pinn, plot_extension)
    cfd_data: np.ndarray = np.array(par_log_parser_pinn.get_cfd_arr(convert_var_name(field)))
    pinn_data: np.ndarray = np.array(par_log_parser_pinn.get_pinn_arr(convert_var_name(field)))

    par_log_parser_ann: HistLogParser = HistLogParser(model_dir_ann, plot_extension)
    ann_data: np.ndarray = np.array(par_log_parser_ann.get_pinn_arr(convert_var_name(field)))

    create_error_scatter_plot(axis=ax4,
                              cfd_data=cfd_data[::75],
                              nn_data_dict={"PINN": pinn_data[::75], "ANN": ann_data[::75]},
                              field=field
                              )

    ax4.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncols=2,
        frameon=False
    )

    plt.subplots_adjust(wspace=0.2)

    save_plot(fig=fig, path=save_path, filename=f"Combiplot-{convert_var_for_fig_name(field)}.pdf")

if __name__ == "__main__":
    model_dir_pinn: str = filedialog.askdirectory(title="Chosse a PINN model to create plots for.")
    model_dir_ann: str = filedialog.askdirectory(title="Chosse an ANN model to create plots for.")
    fig_dir_name: str = "figures"
    plot_extension: str = ""
    if not os.path.exists(os.path.join(model_dir_pinn, fig_dir_name)):
        fig_dir_name = os.path.basename(filedialog.askdirectory(title="Choose the directory of the figures."))
        plot_extension = fig_dir_name.split("_", maxsplit=1)[1]

    color_log_parser_pinn: ColorLogParser = ColorLogParser(model_dir_pinn, plot_extension)
    color_log_parser_ann: ColorLogParser = ColorLogParser(model_dir_ann, plot_extension)
    plot_combination(file_path_pinn=color_log_parser_pinn.get_log_path(),
                     file_path_ann=color_log_parser_ann.get_log_path(),
                     x_info=color_log_parser_pinn.get_flow_plane()[0],
                     y_info=color_log_parser_pinn.get_flow_plane()[1],
                     field=convert_var_name_rev("eta"),
                     save_path=os.path.join(model_dir_pinn, fig_dir_name)
                     )