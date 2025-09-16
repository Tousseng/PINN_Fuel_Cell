from __future__ import annotations

from postprocessing.loss_plots.settings import *
from matplotlib.pyplot import Axes
from matplotlib.collections import PathCollection

def _create_loss_over_time(loss_histories: dict[str,list[float]],
                           iter_time_history: dict[str,dict[str,float]],
                           varying_param: str,
                           ax: Axes,
                           x_positions: Iterator,
                           use_log: bool = False) -> None:

    ax.set_prop_cycle(
        cycler(color=color_list_for_plot(color_scheme[varying_param]))
    )
    scatters: list[PathCollection] = []
    marker: str = next(markers_iter) if varying_param != "-42" else "*"
    for val, loss_history in loss_histories.items():
        final_loss: float = loss_history[-1]
        final_time: float = iter_time_history["Adam"][val] + iter_time_history["LBFGS"][val]
        scatters.append(
            ax.scatter(
                x=final_time, y=final_loss, s=8 if varying_param != "-42" else 12, label=val,
                marker=marker, zorder=3, edgecolors="black", linewidths=0.5
            )
        )

    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Loss / -")
    if use_log:
        ax.set_yscale("log")

    if varying_param != "-42":
        leg = ax.legend(
            title=f"{varying_param.replace('_', ' ').capitalize()}",
            handles=scatters,
            loc="upper center",
            bbox_to_anchor=(next(x_positions), 1.7),
            frameon=False
        )
        ax.add_artist(leg)

def plot_loss_over_time(dir_path: str, use_log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=figsize_oc)

    varying_params: list[str] = []
    x_positions: Iterator[float] = iter([0.05, 0.25, 0.5, 0.73, 0.9])

    for curr_var in os.listdir(dir_path):
        if curr_var.endswith("pdf") or curr_var.endswith("svg"):
            continue
        curr_var_path: str = os.path.join(dir_path, curr_var)
        loss_log_parser: LossLogParser = LossLogParser(curr_var_path)
        if "var" in curr_var:
            varying_param: str = loss_log_parser.get_varying_param()
            loss_histories: dict[str,list[float]] = loss_log_parser.get_loss_history(train_info=True)
            iter_time_history: dict[str,dict[str,float]] = loss_log_parser.get_iter_time_history()
        else:
            varying_param = "-42"
            loss_histories = {varying_param: loss_log_parser.get_singular_loss_history(train_info=True)}
            iter_time_history = {
                "Adam": {varying_param: loss_log_parser.get_adam_train_time()},
                "LBFGS": {varying_param: loss_log_parser.get_lbfgs_train_time()}
            }

        varying_params.append(varying_param)
        _create_loss_over_time(
            loss_histories=loss_histories,
            iter_time_history=iter_time_history,
            varying_param=varying_param,
            ax=ax,
            x_positions=x_positions,
            use_log=use_log
        )
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
    fig.legend(
        handles=[],
        bbox_to_anchor=(0.5,1.4),
        frameon=False
    )

    save_plot(fig, dir_path, f"loss_over_time.pdf")
    save_plot(fig, dir_path, f"loss_over_time.svg")

if __name__ == "__main__":
    from tkinter import filedialog
    dir_path: str = filedialog.askdirectory(title="Choose folder of the model to create the plots.")

    plot_loss_over_time(dir_path, use_log=True)