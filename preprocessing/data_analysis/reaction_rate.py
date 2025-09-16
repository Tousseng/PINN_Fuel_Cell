import numpy as np

from preprocessing.data_analysis.utils import *

def calc_tafel_values(data: pd.DataFrame, pres_O2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # At the end, the additional factor of '2.1' can be seen.
    j_0: np.ndarray = 3.3232 * pres_O2 / 101300 * np.exp(-72400 / 8.314 * (1 / 353.15 - 1 / 298.15)) * 2.1

    alph_a = 0.62
    alph_c = 1 - alph_a

    model: np.ndarray = (
            np.log(j_0) + 0.55 * np.log(select_data(data, "c_O2")) +
            alph_a * 96485.339 * select_data(data, "eta") / (8.314 * 353.15)
    )

    res_curr_dens: np.ndarray[float] = np.log(select_data(data, "j")) - model

    return res_curr_dens, model, j_0

def calc_butler_volmer_values(data: pd.DataFrame, pres_O2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j_0: np.ndarray = 3.3232 * pres_O2 / 101300 * np.exp(-72400 / 8.314 * (1 / 353.15 - 1 / 298.15)) * 1.98565

    alph_a = 0.67
    alph_c = 1 - alph_a

    delta_a: np.ndarray = np.exp(alph_a * 96485.339 / (8.314 * 353.15))
    delta_c: np.ndarray = np.exp(- alph_c * 96485.339 / (8.314 * 353.15))

    model: np.ndarray[float] = (j_0 * (select_data(data, "c_O2")) ** 0.55 *
                                (delta_a ** select_data(data, "eta")) -
                                delta_c ** select_data(data, "eta")
                                )

    res_curr_dens: np.ndarray[float] = select_data(data, "j") - model

    return res_curr_dens, model, j_0

def reaction_rate(show_plots: bool = True):
    data: pd.DataFrame = load_data().reset_index()

    arr_x: np.ndarray = select_data(data, "z")
    arr_y: np.ndarray = select_data(data, "y")

    m: np.ndarray = (1 / (select_data(data, "Y_H2") / 2.016 +
                          select_data(data, "Y_O2") / 31.999 +
                          select_data(data, "Y_H2O") / 18.0155 +
                          select_data(data, "Y_N2") / 28.013)
                     )

    pres_O2: np.ndarray = m / 31.999 * select_data(data, "Y_O2") * select_data(data, "p")

    # Tafel equation
    res_curr_dens, model, j_0 = calc_tafel_values(data=data, pres_O2=pres_O2)
    model = np.exp(model)
    res_curr_dens = np.exp(res_curr_dens)
    # Tafel equation

    # Butler-Volmer equation
    # res_curr_dens, model, j_0 = calc_butler_volmer_values(data=data, pres_O2=pres_O2)
    # Butler-Volmer equation

    grid_x, grid_y, grid_res_curr_dens = to_even_grid(arr_x, arr_y, res_curr_dens)
    _, _, grid_model = to_even_grid(arr_x, arr_y, model)
    _, _, grid_eta = to_even_grid(arr_x, arr_y, select_data(data, "eta"))
    _, _, grid_j_0 = to_even_grid(arr_x, arr_y, j_0)
    _, _, grid_m = to_even_grid(arr_x, arr_y, m)

    print_stats(res_curr_dens, "res_curr_dens")
    print_stats(m, "mol_mass")
    print_stats(j_0, "j_0")
    print_stats(model, "model")

    plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_res_curr_dens,
        var=r"$\Delta j$", var_unit=r"$\mathrm{A/m}^2$"
    )

    plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_model,
        var=r"Model $j$", var_unit=r"$\mathrm{A/m}^2$"
    )

    plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_j_0,
        var=r"$j_0$", var_unit=r"$\mathrm{A/m}^2$"
    )

    plot(
        grid_x=grid_x, grid_y=grid_y, grid_field=grid_m,
        var=r"$M$", var_unit=r"$\mathrm{g/mol}$"
    )

    if show_plots:
        plt.show()

if __name__ == "__main__":
    reaction_rate()