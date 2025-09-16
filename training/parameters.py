import numpy as np
import tensorflow as tf

### TOGGLE THE TYPE OF PART THAT IS INVESTIGATED ###
part: str = ""

is_cathode: bool = True

### PARAMETERS FOR NON-DIMENSIONAL PDEs ###
vel_refs: dict[str,tf.Tensor] = {
    "x": tf.constant(1e-4, dtype=tf.float32),
    "y": tf.constant(1e-3, dtype=tf.float32),
    "z": tf.constant(1e1, dtype=tf.float32)
}
pos_refs: dict[str,tf.Tensor] = {
    "x": tf.constant(1e-4, dtype=tf.float32), # Investigate if this is the correct dimension.
    "y": tf.constant(1e-3, dtype=tf.float32),
    "z": tf.constant(50e-3, dtype=tf.float32)
}

# Species Transport
### PARAMETERS FOR NON-DIMENSIONAL PDEs ###
### GENERAL PARAMETERS ###
# Porosity and tortuosity
porosity_gdl: tf.Tensor = tf.constant(0.65, dtype=tf.float32) # porosity of GDL
porosity_mpl: tf.Tensor = tf.constant(0.58, dtype=tf.float32)  # porosity of MPL
coeff: tf.Tensor = tf.constant(-2.97, dtype=tf.float32) # GDL_MPL_Coefficient
# Porosity and tortuosity

# Coefficient for water permeability
water_perm: tf.Tensor = tf.maximum(tf.constant(8.7e-12,dtype=tf.float32),
                                   tf.constant(1e-20,dtype=tf.float32)
                                   ) # m^2
# Coefficient for water permeability

# Molar masses -> All lower case
mol_mass_H2: tf.Tensor = tf.constant(2.016, dtype=tf.float32) #kg/kmol
mol_mass_O2: tf.Tensor = tf.constant(31.999, dtype=tf.float32) #kg/kmol
mol_mass_H2O: tf.Tensor = tf.constant(18.0155, dtype=tf.float32) #kg/kmol
mol_mass_N2: tf.Tensor = tf.constant(28.013, dtype=tf.float32) #kg/kmol
# Molar masses -> All lower case

# Material data: Cathode -> O2, N2, and H2O
rho_c: tf.Tensor = tf.constant(1.793, dtype=tf.float32) # kg/m^3 -> Only GC average (Final_Channel_Design)
mu_c: tf.Tensor = tf.constant(1.759e-5, dtype=tf.float32) # Pa s -> Only GC average (Final_Channel_Design)
diff_c: tf.Tensor = tf.constant(9.809e-6, dtype=tf.float32)  # m²/s -> Average of GC and CL averages (Final_Channel_Design)
# Material data: Cathode -> O2, N2, and H2O

# Material data: Anode -> H2 and N2
rho_a: tf.Tensor = tf.constant(0.541, dtype=tf.float32) # kg/m^3 -> Only GC average (Final_Channel_Design)
mu_a: tf.Tensor = tf.constant(1.369e-5, dtype=tf.float32) # Pa s -> Only GC average (Final_Channel_Design)
diff_a: tf.Tensor = tf.constant(2.493e-5, dtype=tf.float32) # m²/s -> Average of GC and CL averages (Final_Channel_Design)
# Material data: Anode -> H2 and N2

rho: tf.Tensor = rho_c if is_cathode else rho_a
mu: tf.Tensor = mu_c if is_cathode else mu_a
diff: tf.Tensor = diff_c if is_cathode else diff_a

vel_ref: tf.Tensor = vel_refs["z"]
pos_ref: tf.Tensor = pos_refs["z"]
massfrac_ref: tf.Tensor = tf.constant(1e-2, dtype=tf.float32)
press_ref: tf.Tensor = rho * tf.pow(vel_ref, 2)
press_diff: tf.Tensor = tf.constant(3e3, dtype=tf.float32)

reynolds: tf.Tensor = rho * vel_ref * pos_ref / mu
peclet: tf.Tensor = vel_ref * pos_ref / diff

# PARAMETERS FOR THE MOMENTUM BALANCE #
mom_press_corr: tf.Tensor = tf.constant(1.0,dtype=tf.float32)
#mom_press_corr: tf.Tensor = tf.divide(press_diff, press_ref)
mom_diff_corr: tf.Tensor = tf.constant(1.0,dtype=tf.float32)
mom_press_loss: tf.Tensor = tf.constant(0,dtype=tf.float32)
press_loss_coeff = tf.divide(mu, water_perm)  # kg/(m³ s)
# PARAMETERS FOR THE MOMENTUM BALANCE #

# PARAMETERS FOR THE SPECIES TRANSPORT #
spec_diff_corr: tf.Tensor = tf.constant(1,dtype=tf.float32)
# PARAMETERS FOR THE SPECIES TRANSPORT #

# PARAMETERS FOR THE REACTION RATE #
temp: tf.Tensor = tf.constant(353.15, dtype=tf.float32) # K (Star-CCM+)
temp_ref: tf.Tensor = tf.constant(298.15, dtype=tf.float32) # K (Star-CCM+)
alpha_a_cat: tf.Tensor = tf.constant(0.62, dtype=tf.float32) # 1 (Star-CCM+)
alpha_c_cat: tf.Tensor = tf.constant(0.38, dtype=tf.float32) # 1 (Star-CCM+)
alpha_a_ano: tf.Tensor = tf.constant(0.5, dtype=tf.float32) # 1 (Star-CCM+)
alpha_c_ano: tf.Tensor = tf.constant(0.5, dtype=tf.float32) # 1 (Star-CCM+)
alpha_a: tf.Tensor = alpha_a_cat if is_cathode else alpha_a_ano # Coefficient of anode reaction of Butler Volmer equation
alpha_c: tf.Tensor = alpha_c_cat if is_cathode else alpha_c_ano # Coefficient for cathode reaction of Butler Volmer equation
gamma_O2: tf.Tensor = tf.constant(0.55, dtype=tf.float32) # 1 (Channel_Model_Documentation)
gamma_H2: tf.Tensor = tf.constant(0.5, dtype=tf.float32) # 1 (Channel_Model_Documentation)
gamma_conc: tf.Tensor = gamma_O2 if is_cathode else gamma_H2 # Exponent for the concentration term of the exchange current density
gamma_c: tf.Tensor = tf.constant(1, dtype=tf.float32) # 1 (Channel_Model_Documentation)
gamma_a: tf.Tensor = tf.constant(0.5, dtype=tf.float32) # 1 (Channel_Model_Documentation)
gamma: tf.Tensor = gamma_c if is_cathode else gamma_a # Exponent for the pressure term of the current density
exchg0_ref_c: tf.Tensor = tf.constant(3.3232, dtype=tf.float32) # A/m^2 (Star-CCM+)
exchg0_ref_a: tf.Tensor = tf.constant(8.0e4, dtype=tf.float32) # A/m^2 (Star-CCM+)
exchg0_ref: tf.Tensor = exchg0_ref_c if is_cathode else exchg0_ref_a
act_energy_c: tf.Tensor = tf.constant(72.4e3, dtype=tf.float32) # J/mol (Channel_Model_Documentation)
act_energy_a: tf.Tensor = tf.constant(16.9e3, dtype=tf.float32) # J/mol (Channel_Model_Documentation)
act_energy: tf.Tensor = act_energy_c if is_cathode else act_energy_a # Activation energy of the exchange current density
p_ref: tf.Tensor = tf.constant(1.013e5, dtype=tf.float32) # Pa (Star-CCM+)
faraday: tf.Tensor = tf.constant(96485.339924, dtype=tf.float32) # C/mol (Star-CCM+)
gas_const: tf.Tensor = tf.constant(8.3144621, dtype=tf.float32) # J/(mol K) (Star-CCM+)
mem_area: tf.Tensor = tf.constant(3.1114e-4, dtype=tf.float32) # m^2 (Star-CCM+)

channel_widths_ref: tf.Tensor = tf.constant([
    4e-4,
    5e-4,
    6e-4,
    6.87388e-4,
    7e-4,
    8e-4,
    10e-4,
    11e-4
], dtype=tf.float32)

resist_coeffs: tf.Tensor = tf.constant([
    [
        +2.76898093e+15,
        -1.51495140e+12,
        -1.31711777e+09,
        +1.69314798e+06,
        -9.85930864e+02,
        +5.82312024e-01
    ],[
        +1.68754465e+15,
        -1.12139495e+12,
        -8.80116928e+08,
        +1.22307006e+06,
        -5.38962396e+02,
        +4.44783212e-01
    ],[
        +1.21415834e+15,
        -9.86853937e+11,
        -7.44716147e+08,
        +1.00250420e+06,
        -2.48220767e+02,
        +3.71842443e-01
    ],[
        +8.64744578e14,
        -1.06028635e12,
        -5.68186036e8,
        +9.52843858e5,
        -1.14282870e2,
        +2.70007141e-1
    ],[
        +1.11164587e+15,
        -8.86318163e+11,
        -8.05160160e+08,
        +8.43310367e+05,
        -1.93119690e+01,
        +3.35756103e-01
    ],[
        +1.14366483e+15,
        -7.37359582e+11,
        -9.30252751e+08,
        +6.83902270e+05,
        +1.70571364e+02,
        +3.23970437e-01
    ],[
        +8.92347961e+14,
        -3.49723869e+11,
        -9.30274269e+08,
        +3.51912309e+05,
        +4.25263459e+02,
        +3.50423345e-01
    ],[
        +6.44801872e+14 ,
        -2.13204872e+11,
        -8.05319343e+08,
        +2.25704794e+05,
        +4.99855110e+02,
        +3.79098561e-01
    ]
], dtype=tf.float32)

@tf.function
def lookup_coeffs(channel_widths: tf.Tensor) -> tf.Tensor:
    # Find the reference channel widths for all supplied channel_widths
    cmp = tf.less_equal(
        tf.abs(tf.expand_dims(channel_widths, -1) - tf.expand_dims(channel_widths_ref, 0)), 2e-5
    )
    # Get index of first hit; index is then used to get the coefficients from "resist_coeffs"
    idx = tf.argmax(tf.cast(cmp, tf.int32), axis=1)

    return tf.gather(resist_coeffs, idx)

@tf.function
def calc_resist(y: tf.Tensor, channel_widths: tf.Tensor) -> tf.Tensor:
    coeffs: tf.Tensor = lookup_coeffs(channel_widths)

    a, b, c, d, e, f = tf.unstack(coeffs, axis=1)

    return (
            a * tf.pow(y, 5) + b * tf.pow(y, 4) + c * tf.pow(y, 3) + d * tf.pow(y, 2) + e * y + f
    )
# PARAMETERS FOR THE REACTION RATE #

# CHANGE PARAMETERS BASED ON INVESTIGATED PART #
if part == "GDL":
    mom_press_corr = tf.divide(mom_press_corr, tf.square(porosity_gdl))
    mom_diff_corr = tf.divide(mom_diff_corr, tf.square(porosity_gdl))
    mom_press_loss = press_loss_coeff * pos_ref / (rho * vel_ref) / porosity_gdl
    spec_diff_corr = tf.pow(porosity_gdl, - coeff)
elif part == "MPL":
    mom_press_corr = tf.divide(mom_press_corr, tf.square(porosity_mpl))
    mom_diff_corr = tf.divide(mom_diff_corr, tf.square(porosity_mpl))
    mom_press_loss = press_loss_coeff * pos_ref / (rho * vel_ref) / porosity_mpl
    spec_diff_corr = tf.pow(porosity_mpl, - coeff)
# CHANGE PARAMETERS BASED ON INVESTIGATED PART #