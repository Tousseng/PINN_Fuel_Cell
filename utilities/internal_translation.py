in_trans_rev: dict[str,str] = {
    "x": "X (m)",
    "y": "Y (m)",
    "z": "Z (m)"
}

out_trans_rev: dict[str,str] = {
    "rho": "Density (kg/m^3)",
    "p": "Pressure (Pa)",
    "u": "Velocity[i] (m/s)",
    "v": "Velocity[j] (m/s)",
    "w": "Velocity[k] (m/s)",
    "Y_O2": "Mass Fraction of O2 of Gas",
    "j": "Electrochemical Reaction Rate of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (A/m^2)",
    "c_O2": "Molar Concentration of O2 of Gas (kmol/m^3)",
    "eta": "Electrochemical Surface Overpotential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)",
    "U_eq": "Electrochemical Equilibrium Potential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)",
    "Y_H2": "Mass Fraction of H2 of Gas",
    "Y_H2O": "Mass Fraction of H2O of Gas",
    "Y_N2": "Mass Fraction of N2 of Gas",
    "mu": "Dynamic Viscosity (Pa-s)",
    "D_H2": "Effective Mass Diffusivity of H2 of Gas (m^2/s)",
    "D_O2": "Effective Mass Diffusivity of O2 of Gas (m^2/s)"
}

pred_trans_rev: dict[str,str] = {"b": "Channel Width (m)"}

in_trans: dict[str,str] = {
    "X (m)": "x",
    "Y (m)": "y",
    "Z (m)": "z"
}

out_trans: dict[str,str] = {
    "Density (kg/m^3)": "rho",
    "Pressure (Pa)": "p",
    "Velocity[i] (m/s)": "u",
    "Velocity[j] (m/s)": "v",
    "Velocity[k] (m/s)": "w",
    "Mass Fraction of O2 of Gas": "Y_O2",
    "Electrochemical Reaction Rate of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (A/m^2)": "j",
    "Molar Concentration of O2 of Gas (kmol/m^3)": "c_O2",
    "Electrochemical Surface Overpotential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)": "eta",
    "Electrochemical Equilibrium Potential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)": "U_eq",
    "Mass Fraction of H2 of Gas": "Y_H2",
    "Mass Fraction of H2O of Gas": "Y_H2O",
    "Mass Fraction of N2 of Gas": "Y_N2",
    "Dynamic Viscosity (Pa-s)": "mu",
    "Effective Mass Diffusivity of O2 of Gas (m^2/s)": "D_O2"
}

pred_trans: dict[str,str] = {"Channel Width (m)": "b"}

plot_axis_trans: dict[str,str] = {
    "X (m)": r"Channel Height",
    "Y (m)": r"Cell Width",
    "Z (m)": r"Channel Length",
    "Density (kg/m^3)": r"$\rho$",
    "Pressure (Pa)": r"$p$",
    "Velocity[i] (m/s)": r"$u$",
    "Velocity[j] (m/s)": r"$v$",
    "Velocity[k] (m/s)": r"$w$",
    "Mass Fraction of O2 of Gas": r"$Y_\mathrm{O_2}$",
    "Electrochemical Reaction Rate of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (A/m^2)":
        r"$j$",
    "Molar Concentration of O2 of Gas (kmol/m^3)": r"$c_\mathrm{O_2}$",
    "Electrochemical Surface Overpotential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)":
        r"$\eta$",
    "Electrochemical Equilibrium Potential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)":
        r"$U_\mathrm{eq}$",
    "Mass Fraction of H2 of Gas": r"$Y_\mathrm{H_2}$",
    "Mass Fraction of H2O of Gas": r"$Y_\mathrm{H_2O}$",
    "Mass Fraction of N2 of Gas": r"$Y_\mathrm{N_2}$",
    "Channel Width (m)": r"Channel Width",
    "Current Density (A/cm^2)": r"Current Density $j_\mathrm{VC}$"
}

plot_title_trans: dict[str,str] = {
    "Density (kg/m^3)": "Density",
    "Pressure (Pa)": "Pressure",
    "Velocity[i] (m/s)": "Velocity",
    "Velocity[j] (m/s)": "Velocity",
    "Velocity[k] (m/s)": "Velocity",
    "Mass Fraction of O2 of Gas": "Mass Fraction",
    "Electrochemical Reaction Rate of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (A/m^2)":
        "Current Density",
    "Molar Concentration of O2 of Gas (kmol/m^3)": "Molar Concentration",
    "Electrochemical Surface Overpotential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)":
        "Overpotential",
    "Electrochemical Equilibrium Potential of 1/2 O2 + 2 H+ + 2 e- <> H2O of Cathodic of Fluid (V)":
        "Equilibrium Potential",
    "Mass Fraction of H2 of Gas": "Mass Fraction",
    "Mass Fraction of H2O of Gas": "Mass Fraction",
    "Mass Fraction of N2 of Gas": "Mass Fraction",
    "Channel Width (m)": "Channel Width"
}

extra_fig_name: dict[str,str] = {"Velocity[i] (m/s)": "u",
                                 "Velocity[j] (m/s)": "v",
                                 "Velocity[k] (m/s)": "w",
                                 "Mass Fraction of O2 of Gas": "O2",
                                 "Mass Fraction of H2 of Gas": "H2",
                                 "Mass Fraction of H2O of Gas": "H2O",
                                 "Mass Fraction of N2 of Gas": "N2"
                                 }

loss_trans: dict[str,str] = {
    "total": r"$\mathcal{L}_\mathrm{Tot}$",
    "conti": r"$\mathcal{L}_\mathrm{Con}$",
    "momentum": r"$\mathcal{L}_\mathrm{Mom}$",
    "spec_trans": r"$\mathcal{L}_\mathrm{Spe}$",
    "momentum_bc_u": r"$\mathcal{L}_{\mathrm{BC},u}$",
    "bc_u": r"$\mathcal{L}_{\mathrm{BC},u}$",
    "momentum_bc_w": r"$\mathcal{L}_{\mathrm{BC},w}$",
    "bc_w": r"$\mathcal{L}_{\mathrm{BC},w}$",
    "momentum_bc_p": r"$\mathcal{L}_{\mathrm{BC},p}$",
    "bc_p": r"$\mathcal{L}_{\mathrm{BC},p}$",
    "spec_trans_bc": r"$\mathcal{L}_{\mathrm{BC},Y_\mathrm{O_2}}$",
    "bc_Y_O2": r"$\mathcal{L}_{\mathrm{BC},Y_\mathrm{O_2}}$",
    "surf_pot": r"$\mathcal{L}_\mathrm{Surf}$",
    "curr_dens": r"$\mathcal{L}_\mathrm{Rea}$",
    "width": r"$\mathcal{L}_\mathrm{Width}$",
    "total_val": r"$\mathcal{L}_\mathrm{Tot}^\mathrm{val}$",
    "conti_val": r"$\mathcal{L}_\mathrm{Con}^\mathrm{val}$",
    "momentum_val": r"$\mathcal{L}_\mathrm{Mom}^\mathm{val}$",
    "spec_trans_val": r"$\mathcal{L}_\mathrm{Spe}$^\mathrm{val}",
    "momentum_bc_u_val": r"$\mathcal{L}_{\mathrm{BC},u}^\mathrm{val}$",
    "bc_u_val": r"$\mathcal{L}_{\mathrm{BC},u}^\mathrm{val}$",
    "momentum_bc_w_val": r"$\mathcal{L}_{\mathrm{BC},w}^\mathrm{val}$",
    "bc_w_val": r"$\mathcal{L}_{\mathrm{BC},w}^\mathrm{val}$",
    "momentum_bc_p_val": r"$\mathcal{L}_{\mathrm{BC},p}^\mathrm{val}$",
    "bc_p_val": r"$\mathcal{L}_{\mathrm{BC},p}^\mathrm{val}$",
    "spec_trans_bc_val": r"$\mathcal{L}_{\mathrm{BC},Y_\mathrm{O_2}}^\mathrm{val}$",
    "bc_Y_O2_val": r"$\mathcal{L}_{\mathrm{BC},Y_\mathrm{O_2}}^\mathrm{val}$",
    "surf_pot_val": r"$\mathcal{L}_\mathrm{Surf}^\mathrm{val}$",
    "curr_dens_val": r"$\mathcal{L}_\mathrm{Rea}^\mathrm{val}$",
    "width_val": r"$\mathcal{L}_\mathrm{Width}^\mathrm{val}$",
    "data": r"$\mathcal{L}_\mathrm{Data}$"
}

surf_avg_trans: dict[str, str] = {
    "CFD": r"CFD",
    "PINN_best": r"PINN$_\mathrm{best}$",
    "PINN_worst": r"PINN$_\mathrm{worst}$",
    "ANN_best": r"ANN$_\mathrm{best}$",
    "ANN_worst": r"ANN$_\mathrm{worst}$"
}

def convert_var_name(var: str) -> str:
    return (in_trans | out_trans | pred_trans).get(var)

def convert_var_name_rev(var: str) -> str:
    return (in_trans_rev | out_trans_rev | pred_trans_rev).get(var)

def convert_var_for_plot_axis(var: str) -> tuple[str,str]:
    split_var: list[str] = var.split(sep="(")
    unit: str = "-"
    # Field without unit doesn't have second element in split_field.
    if len(split_var) > 1:
        unit = split_var[1].split(sep=")")[0]
    split_unit: list[str] = unit.split(sep="^")
    if len(split_unit) > 1:
        unit = split_unit[0] + r"$^{exp}$".format(exp=split_unit[1])
    return plot_axis_trans.get(var), unit

def convert_var_for_plot_title(var: str) -> str:
    return r"{label} {unit}".format(label=plot_title_trans.get(var), unit=plot_axis_trans.get(var))

def convert_var_for_fig_name(var: str) -> str:
    extra: str = ""
    if var in list(extra_fig_name.keys()):
        extra = f"_{extra_fig_name.get(var)}"
    return f"{plot_title_trans.get(var).lower().replace(' ', '_')}{extra}"

def convert_var_list_rev(variables: list[str]) -> list[str]:
    conv_list: list[str] = []
    for var in variables:
        conv_list.append(convert_var_name_rev(var))

    return conv_list

def convert_surf_avg_name(surf_avg_name: str) -> str:
    return surf_avg_trans.get(surf_avg_name)

def convert_loss_names(loss_name: str) -> str:
    return loss_trans.get(loss_name)

if __name__ == "__main__":
    pass