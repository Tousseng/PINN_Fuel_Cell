from __future__ import annotations

from utilities.internal_translation import convert_loss_names
from postprocessing.loss_plots.settings import *
from utilities import annotate_interval

# Plotting of the loss function
def plot_loss(loss_comps_history: dict[str,list[float]], val_loss_comps_history: dict[str,list[float]],
              adam_iters: int, lbfgs_iters: int, val_adam_iters: int | None,
              val_adam_interval: int | None, val_lbfgs_interval: int | None,
              adam_train_time: float, lbfgs_train_time: float,
              save_path: str,  exclude_comps: tuple[str] = (),
              use_log: bool = False) -> None:
    # Plotting the loss values via the indices (training steps or iterations)
    fig1, ax1 = plt.subplots(figsize=figsize_oc)
    fig1.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)

    for loss_name, loss_vals in loss_comps_history.items():
        if loss_name in exclude_comps:
            continue
        color: tuple[float,...] = color_for_plot(comp_colors[loss_name])
        ax1.plot(range(1, adam_iters + 1), loss_vals[:adam_iters],
                 color=color,
                 label=f"{convert_loss_names(loss_name)}"
                 )
    if val_adam_iters is not None and val_adam_interval is not None:
        val_loss_vals: list[float] = val_loss_comps_history["total"]
        ax1.plot(
            [val_adam_interval*x+1 for x in list(range(val_adam_iters))],
            val_loss_vals[:val_adam_iters], color="red",
            label=f"total"
        )
    ax1.set_xlabel("Iteration / -")
    ax1.set_ylabel("Loss / -")
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if use_log:
        ax1.set_yscale("log")
    ax1.legend(
        loc="best",
        frameon=True,
        title="Loss Comps",
        ncol=3
    )

    # Plotting the loss values over the epochs
    fig2, ax2 = plt.subplots(figsize=figsize_oc)
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
    ax2.set_prop_cycle(  cycler(color=color_list_for_plot(loss_comps_colors))
                       + cycler(linestyle=linestyle_list_for_plot(lines))
                       )
    for loss_name, loss_vals in loss_comps_history.items():
        if loss_name in exclude_comps:
            continue
        color: tuple[float,...] = color_for_plot(comp_colors[loss_name])
        ax2.plot(range(1, len(loss_vals) + 1), loss_vals,
                 color=color,
                 label=f"{convert_loss_names(loss_name)}"
                 )
    if val_adam_iters is not None and val_adam_interval is not None:
        val_loss_vals: list[float] = val_loss_comps_history["total"]
        ax2.plot(
            [val_adam_interval*x+1 for x in list(range(val_adam_iters))] +
            [val_lbfgs_interval*x+adam_iters+1 for x in list(range(len(val_loss_vals) - val_adam_iters))],
            val_loss_vals, color="red",
            label=f"{convert_loss_names('total_val')}"
        )
    ax2.set_xlabel("Iteration / -")
    ax2.set_ylabel("Loss / -")
    ax2.set_xlim(0, xmax=None)
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if use_log:
        ax2.set_yscale("log")
    ax2.legend(
        loc="best",
        frameon=True,
        title="Loss comps",
        ncol=3
    )

    # Insert vertical dashed line at lbfgs_start_index
    if lbfgs_iters > 0:
        ax2.axvline(x=adam_iters+1, color="grey", linestyle="--", linewidth=1)
        textstr = (
                   "Adam " + r"$t_\mathrm{train}$" + f": {adam_train_time:.2f} s,\t" +
                   "L-BFGS " + r"$t_\mathrm{train}$" + f": {lbfgs_train_time:.2f} s" #+
                   )
    else:
        textstr = f"Training duration Adam Optimizer: {adam_train_time:.2f}s"

    ax2.text(0.5, 1.1, textstr, transform=ax2.transAxes,
             verticalalignment="top", horizontalalignment="center", zorder=3
             )

    y_pos: float = 0.83 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])

    annotate_interval(ax=ax2, x_start=ax2.get_xlim()[0], x_end=adam_iters+1, y_pos=y_pos, text="Adam")
    annotate_interval(ax=ax2, x_start=adam_iters+1, x_end=ax2.get_xlim()[1], y_pos=y_pos, text="L-BFGS")

    save_plot(fig1, save_path, "adam_loss.pdf")
    save_plot(fig1, save_path, "adam_loss.svg")
    save_plot(fig2, save_path, "adam_lbfgs_loss.pdf")
    save_plot(fig2, save_path, "adam_lbfgs_loss.svg")
    plt.close(fig="all")

if __name__ == "__main__":
    from tkinter import filedialog

    log_path = filedialog.askdirectory(title="Choose folder of the model to create the plots.")
    loss_log_parser: LossLogParser = LossLogParser(log_path)
    plot_loss(
        loss_comps_history=loss_log_parser.get_loss_comps_history(train_info=True),
        val_loss_comps_history=loss_log_parser.get_loss_comps_history(train_info=False),
        adam_iters=loss_log_parser.get_adam_iters(),
        lbfgs_iters=loss_log_parser.get_lbfgs_iters(),
        val_adam_iters=loss_log_parser.get_adam_val_iters(),
        val_adam_interval=loss_log_parser.get_adam_val_interval(),
        val_lbfgs_interval=loss_log_parser.get_lbfgs_val_interval(),
        adam_train_time=loss_log_parser.get_adam_train_time(),
        lbfgs_train_time=loss_log_parser.get_lbfgs_train_time(),
        save_path=log_path,
        use_log=True
    )