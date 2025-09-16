from postprocessing.loss_plots.settings import *

from utilities import annotate_interval

# Plotting of the loss function
def _plot_variation_loss(loss_histories: dict[str,list[float]], iter_time_history: dict[str,dict[str,float]],
                        varying_param: str, adam_iters: int, save_path: str, use_log: bool, every: int) -> None:
    # Plotting the loss values via the indices (training steps or iterations)
    fig1, ax1 = plt.subplots(figsize=figsize_oc)
    fig1.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
    ax1.set_prop_cycle(  cycler(color=color_list_for_plot(color_scheme[varying_param]))
                       + cycler(linestyle=linestyle_list_for_plot(lines))
                       + cycler(marker=markers)
                       )
    for val, loss_history in loss_histories.items():
        adam_iters: int = int(val) if varying_param == "epochs" else adam_iters
        adam_train_time: float = iter_time_history["Adam"][str(val)]
        ax1.plot(
            range(1,adam_iters+1), loss_history[:adam_iters], markerfacecolor=color_for_plot("White"),
            markevery=every, label=f"{val}; " + r"$t_\mathrm{train}$" + f": {adam_train_time:.1f} s"
        )

    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.set_xlabel("Iteration / -")
    ax1.set_ylabel("Loss / -")
    if use_log:
        ax1.set_yscale("log")
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        ncols=2,
        frameon=False
    )

    # Plotting the loss values over the epochs
    fig2, ax2 = plt.subplots(figsize=figsize_oc)
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
    ax2.set_prop_cycle(cycler(color=color_list_for_plot(color_scheme[varying_param]))
                       + cycler(linestyle=linestyle_list_for_plot(lines))
                       + cycler(marker=markers)
                       )
    all_adam_iters: list[int] = []
    for val, loss_history in loss_histories.items():
        adam_iters: int = int(val) if varying_param == "epochs" else adam_iters
        if varying_param == "epochs":
            all_adam_iters.append(adam_iters)
        lbfgs_iters: int = len(loss_history[adam_iters:])
        total_train_time: float = iter_time_history["Adam"][str(val)] + iter_time_history["LBFGS"][str(val)]
        ax2.plot(
            range(1, len(loss_history) + 1), loss_history, markerfacecolor=color_for_plot("White"),
            markevery=every,
            label=f"{val}; " + r"$t_\mathrm{train}$" + f": {total_train_time:.1f} s"
                 )
    if not all_adam_iters:
        all_adam_iters.append(adam_iters)

    for idx, curr_adam_iters in enumerate(all_adam_iters):
        ax2.axvline(x=curr_adam_iters + 1, color="grey", linestyle="--", linewidth=1)
        if idx == 0 or (idx == (len(all_adam_iters) - 1) and idx != 0):
            y_pos: float = (0.83 - 0.05 * idx) * (ax2.get_ylim()[1] - ax2.get_ylim()[0])
            annotate_interval(ax=ax2, x_start=ax2.get_xlim()[0], x_end=curr_adam_iters + 1, y_pos=y_pos, text="Adam")

    ax2.set_xlabel("Iteration / -")
    ax2.set_ylabel("Loss / -")
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if use_log:
        ax2.set_yscale("log")
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        ncols=2,
        frameon=False
    )

    save_plot(fig1, save_path, f"adam_loss_var_{varying_param}.pdf")
    save_plot(fig1, save_path, f"adam_loss_var_{varying_param}.svg")
    save_plot(fig2, save_path, f"adam_lbfgs_loss_var_{varying_param}.pdf")
    save_plot(fig2, save_path, f"adam_lbfgs_loss_var_{varying_param}.svg")
    plt.close(fig="all")

def plot_variation_loss(log_path: str, use_log: bool = False, every: int = 50) -> None:
    loss_log_parser: LossLogParser = LossLogParser(log_path)
    _plot_variation_loss(loss_histories=loss_log_parser.get_loss_history(train_info=True),
                         iter_time_history=loss_log_parser.get_iter_time_history(),
                         varying_param=loss_log_parser.get_varying_param(),
                         adam_iters=loss_log_parser.get_adam_iters(),
                         save_path=log_path,
                         use_log=use_log,
                         every=every
                         )

if __name__ == "__main__":
    from tkinter import filedialog
    log_path = filedialog.askdirectory(title="Choose folder of the model to create the plots.")
    plot_variation_loss(log_path)