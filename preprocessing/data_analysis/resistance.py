from preprocessing.data_analysis.utils import *

def calc_resistance(data: pd.DataFrame) -> np.ndarray[float]:
    return (
            - (select_data(data, "U_eq") + select_data(data, "eta"))
            / (select_data(data, "j") * 3.1114e-4)
    )

def plot_combined_resistance(save_path: str, filename: str = "Resistance.pdf") -> Figure:
    var = r"$R_\mathrm{el}$"
    var_unit = r"$\Omega$"

    # Training data set on CL
    data: pd.DataFrame = load_data("training").reset_index()
    arr_x: np.ndarray = select_data(data, "z")
    arr_y: np.ndarray = select_data(data, "y")
    resistance: np.ndarray[float] = calc_resistance(data)
    grid_x, grid_y, grid_resistance = to_even_grid(arr_x, arr_y, resistance)

    # Entire data set on CL
    data_all: pd.DataFrame = load_data("entire").reset_index()
    arr_x_all: np.ndarray = select_data(data_all, "z")
    arr_y_all: np.ndarray = select_data(data_all, "y")
    resistance_all: np.ndarray[float] = calc_resistance(data_all)
    grid_x_all, grid_y_all, grid_resistance_all = to_even_grid(arr_x_all, arr_y_all, resistance_all)

    fig = plt.figure(figsize=figsize_oc_large)
    gs = gridspec.GridSpec(2, 2, width_ratios=[6.5, 3.5], hspace=0.1, figure=fig)

    axis1 = fig.add_subplot(gs[0, 0])
    axis2 = fig.add_subplot(gs[0, 1])
    axis3 = fig.add_subplot(gs[1, :])

    # Entire data set on CL
    _, _, vmin, vmax = plot(
        grid_x=grid_x_all, grid_y=grid_y_all, grid_field=grid_resistance_all, var=var, var_unit=var_unit, use_log=False,
        fig=fig, axis=axis3
    )

    axis3.axhline(y=(arr_y.max() * 10 ** 3 - 0.687388), color="black", linestyle="-")
    axis3.axvline(x=125, color="black", linestyle="-.")

    # Training data set on CL
    plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_resistance, var=var, var_unit=var_unit, use_log=False,
        fig=fig, axis=axis1, vmin=vmin, vmax=vmax, show_colorbar=False, label_top=True
    )

    axis1.axhline(y=(arr_y.max() * 10 ** 3 - 0.687388), color="black", linestyle="-")
    axis1.axvline(x=125, color="black", linestyle="-.")

    # Training daa set on CL
    plot_1D_subset(
        arr_x=arr_x, arr_y=arr_y, arr_field=resistance, pos_x=0.125, fig=fig, axis=axis2
    )

    save_plot(
        fig=fig,
        path=save_path,
        filename=filename
    )

    return fig

def print_resist_model_coeffs() -> None:
    data: pd.DataFrame = load_data("training").reset_index()
    arr_x: np.ndarray = select_data(data, "z")
    arr_y: np.ndarray = select_data(data, "y")
    resistance: np.ndarray[float] = calc_resistance(data)

    coeffs, _ = plot_1D_subset(
        arr_x=arr_x, arr_y=arr_y, arr_field=resistance, pos_x=0.125
    )

    print(f"Coefficients for fifth-order polynomial:\n{coeffs}")

def surface_overpotential(show_figs: bool = True):
    data: pd.DataFrame = load_data().reset_index()

    arr_x: np.ndarray = select_data(data, "z")
    arr_y: np.ndarray = select_data(data, "y")

    # Surface Overpotential

    coeffs, fig_1d_resist = plot_1D_subset(
        arr_x, arr_y, calc_resistance(data), 0.125
    )
    # Do not show resist figure -> use 'plot_combined_resistance()' for that
    plt.close(fig_1d_resist)

    resist: np.ndarray = np.polyval(coeffs, arr_y)

    volt_drop: np.ndarray[float] = resist * 3.1114e-4 * select_data(data, "j")

    res_surf_pot: np.ndarray[float] = (
            select_data(data, "eta") +
            select_data(data, "U_eq") +
            volt_drop
    )

    grid_x, grid_y, grid_res_surf_pot = to_even_grid(arr_x, arr_y, res_surf_pot)
    _, _, grid_eta = to_even_grid(arr_x, arr_y, select_data(data, "eta"))
    _, _, grid_U_eq = to_even_grid(arr_x, arr_y, select_data(data, "U_eq"))
    _, _, grid_j = to_even_grid(arr_x, arr_y, volt_drop)

    print_stats(res_surf_pot, "Surface Overpotential")
    print_stats(select_data(data, "eta"), "eta")
    print_stats(select_data(data, "U_eq"), "U_eq")
    print_stats(select_data(data, "j"), "j")
    print_stats(volt_drop, "j Voltage Vrop")

    fig_eta, _, _, _ = plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_eta,
        var=r"eta"
    )

    fig_U_eq, _, _, _ = plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_U_eq,
        var=r"U_eq"
    )

    fig_volt_drop, _, _, _ = plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_j,
        var=r"$j$ Voltage Drop", var_unit=r"$\mathrm{V}$"
    )

    fig_surf_pot, _, _, _ = plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_res_surf_pot,
        var=r"Surface Overpotential", var_unit=r"$\mathrm{V}$"
    )

    # Surface Overpotential

    if show_figs:
        plt.show()

if __name__ == "__main__":
    #print_resist_model_coeffs()
    plot_combined_resistance(save_path=filedialog.askdirectory(title="Choose the directory to save the plot in."))
    #surface_overpotential()