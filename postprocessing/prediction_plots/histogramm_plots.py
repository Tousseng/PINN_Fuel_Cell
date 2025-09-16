from postprocessing.prediction_plots.settings import *

from scipy.stats import norm

def create_error_histogram_plot(axis: Axes, diff: list[float], mean_diff: float, mean_percent_deviation: float,
                                mse: float, std_diff: float, field: str):
    label, unit = convert_var_for_plot_axis(field)
    _, diff = convert(unit, np.array(diff))
    _, mean_diff = convert(unit, np.array(mean_diff))
    _, mse = convert(unit, np.array(mse))
    unit, std_diff = convert(unit, np.array(std_diff))

    count, bins = np.histogram(diff, bins=30, density=False)

    # Calculate min and max of the bins
    bin_min = bins.min()
    bin_max = bins.max()

    # Left y-axis: PDF (probability density function)
    axis.set_xlabel(f'Difference in {label} / {unit}')
    axis.set_ylabel('Probability Density / -', color=color_for_plot('Black'))
    axis.tick_params(axis='y', labelcolor=color_for_plot('Black'))

    # Calculate the probability density function (PDF)
    x = np.linspace(bin_min, bin_max, 100)
    pdf = norm.pdf(x, mean_diff, std_diff)

    # Right y-axis: Absolute frequency
    axis2 = axis.twinx()
    axis2.set_ylim(count.min(), count.max())
    axis2.set_ylabel('Absolute Data Points / -', color=color_for_plot('DarkBlue'))
    axis2.tick_params(axis='y', labelcolor=color_for_plot('DarkBlue'))

    # Right y-axis (second axis on the right): Percentage frequency
    axis3 = axis.twinx()
    axis3.spines["right"].set_position(("outward", 60))  # Moves the axis outwards
    axis3.set_ylabel('Percentage / %', color=color_for_plot('DarkBlue'))
    axis3.tick_params(axis='y', labelcolor=color_for_plot('DarkBlue'))

    # Set font for tick labels (numerical values on the axes)
    plt.xticks()  # Font and size for X-axis
    plt.yticks()  # Font and size for Y-axis

    # Convert the histogram to percentage frequency and plot it on the right y-axis
    count_percent = (count / np.sum(count)) * 100
    axis3.bar(bins[:-1], count_percent, width=np.diff(bins), edgecolor=color_for_plot('Black'), align="edge", alpha=0.6,
            color=color_for_plot('DarkBlue'), zorder=1)

    # Plot the PDF on the left y-axis
    axis.plot(x, pdf, color=color_for_plot('Black'), linewidth=2, zorder=2)

    axis.set_xlim(bin_min, bin_max)

    # Text with mean_diff, std_diff, mean_percent_deviation and mse
    textstr = (r'$\mu={:.2f}$'.format(mean_diff) + f' {unit},    ' +
               r'$\sigma={:.2f}$'.format(std_diff) + f' {unit}')

    axis.text(0.5, 1.1, textstr, transform=axis.transAxes, wrap=True,
              verticalalignment='top', horizontalalignment="center", zorder=3)

def plot_error_histogram(model_dir: str, fields: list[str], fig_dir_name: str, plot_extension: str = ""):
    for field in fields:
        conv_field: str = convert_var_name(field)
        hist_log_parser: HistLogParser = HistLogParser(model_dir, plot_extension)
        fig, axis = plt.subplots(figsize=figsize_oc, constrained_layout=True)

        create_error_histogram_plot(axis=axis,
                                    diff=hist_log_parser.get_diff(conv_field),
                                    mean_diff=hist_log_parser.get_mean_deviation(conv_field),
                                    mean_percent_deviation=hist_log_parser.get_mean_percent_deviation(conv_field),
                                    mse=hist_log_parser.get_mse(conv_field),
                                    std_diff=hist_log_parser.get_std_deviation(conv_field),
                                    field=field
                                    )
        save_plot(fig=fig, path=os.path.join(model_dir, fig_dir_name),
                  filename=f"Histogram-{convert_var_for_fig_name(field)}.pdf")
        plt.close()

if __name__ == "__main__":
    model_dir = filedialog.askdirectory(title="Choose the directory of the model you want plots for.")
    fig_dir_name: str = "figures"
    plot_extension: str = ""
    if not os.path.exists(os.path.join(model_dir, fig_dir_name)):
        fig_dir_name = os.path.basename(filedialog.askdirectory(title="Choose the directory of the figures."))
        plot_extension = fig_dir_name.split("_", maxsplit=1)[1]
    data_log_parser: DataLogParser = DataLogParser(model_dir)
    plot_error_histogram(model_dir=model_dir,
                         fields=data_log_parser.get_outputs(),
                         fig_dir_name=fig_dir_name,
                         plot_extension=plot_extension
                         )