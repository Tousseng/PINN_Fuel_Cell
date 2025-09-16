from __future__ import annotations

from training.pinn import PINN
from .parameters import *
from utilities.internal_translation import convert_var_name_rev

from typing import Callable

import tensorflow as tf

class Training:
    def __init__(self, pinn: PINN, in_indices: dict[str,int], out_indices: dict[str,int], equations: dict[str,bool],
                 norm_stats: dict[str,dict[str,float]], val_norm_stats: dict[str, dict[str, float]],
                 is_pinn: bool, single_width: float | None, debug: bool):
        self._pinn: PINN = pinn
        self._in_indices: dict[str,int] = in_indices
        self._out_indices: dict[str,int] = out_indices
        self._equations: dict[str,bool] = equations
        self._norm_stats: dict[str,dict[str,float]] = norm_stats
        self._val_norm_stats: dict[str,dict[str,float]] | None = val_norm_stats
        self._outs: dict[str,tf.Tensor] = {}
        self._first_derivs: dict[str,dict[str,tf.Tensor]] = {}
        self._sec_derivs: dict[str,dict[str,tf.Tensor]] = {}
        self._loss_comps: dict[str,float] = {}
        self._is_pinn: bool = is_pinn
        self._single_width: float | None = single_width
        self._debug: bool = debug
        self._terms_and_scales: dict[str,float] = {}

    def get_loss_comps(self) -> dict[str,float]:
        return self._loss_comps

    def get_in_indices(self) -> dict[str,int]:
        return self._in_indices

    def _update_outputs(self, inputs: tf.Tensor) -> None:
        for var, idx in self._out_indices.items():
            self._outs[var] = self._pinn(inputs)[:, idx]

    def _denormalize_var(self, var: str, use_vstats: bool) -> tf.Tensor:
        norm_stats: dict[str,dict[str,float]] = self._norm_stats
        if use_vstats:
            if self._val_norm_stats is None:
                raise ValueError("'self._val_norm_stats' may not be 'None' if 'use_vstats = True'.")
            norm_stats = self._val_norm_stats
        return (
                tf.convert_to_tensor(norm_stats[convert_var_name_rev(var)]["mean"], dtype=tf.float32)
                + tf.convert_to_tensor(norm_stats[convert_var_name_rev(var)]["std"], dtype=tf.float32) *
                self._outs[var]
        )

    def _denormalize(self, vals: tf.Tensor, var: str, use_vstats: bool) -> tf.Tensor:
        norm_stats: dict[str, dict[str, float]] = self._norm_stats
        if use_vstats:
            if self._val_norm_stats is None:
                raise ValueError("'self._val_norm_stats' may not be 'None' if 'use_vstats = True'.")
            norm_stats = self._val_norm_stats
        return (
                tf.convert_to_tensor(norm_stats[convert_var_name_rev(var)]["mean"], dtype=tf.float32)
                + tf.convert_to_tensor(norm_stats[convert_var_name_rev(var)]["std"], dtype=tf.float32) *
                vals[:, (self._out_indices | self._in_indices)[var]]
        )

    @tf.function
    def _calc_bc_loss(self, boundary_inputs: tf.Tensor, boundary_outputs: tf.Tensor, var: str,
                      use_vstats: bool, normalized: bool = False) -> tf.Tensor:
        if normalized:
            ins = self._pinn(boundary_inputs)[:, self._out_indices[var]] # tf.Tensor
            outs = boundary_outputs[:, self._out_indices[var]] # tf.Tensor
        else:
            ins = self._denormalize(self._pinn(boundary_inputs), var, use_vstats) # tf.Tensor
            outs = self._denormalize(boundary_outputs, var, use_vstats) # tf.Tensor
        return outs - ins

    @tf.function
    def _get_var_std(self, var: str, use_vstats: bool) -> tf.Tensor:
        norm_stats: dict[str, dict[str, float]] = self._norm_stats
        if use_vstats:
            if self._val_norm_stats is None:
                raise ValueError("'self._val_norm_stats' may not be 'None' if 'use_vstats = True'.")
            norm_stats = self._val_norm_stats
        return tf.convert_to_tensor(norm_stats[convert_var_name_rev(var)]["std"], dtype=tf.float32)

    @staticmethod
    @tf.function
    def _scale_loss_comp(loss_comp: tf.Tensor, scale: float) -> tf.Tensor:
        return (
            tf.reduce_mean(tf.square(loss_comp)) / tf.constant(scale, dtype=tf.float32)
        )

    @tf.function
    def find_weight_over_threshold(self, grads: tf.Tensor, threshold: float = 5) -> None:
        for var, grad in zip(self._pinn.trainable_variables, grads):
            max_val = tf.reduce_max(tf.abs(var))
            flat_idx = tf.argmax(tf.reshape(var, [-1]))
            pos = tf.unravel_index(flat_idx, tf.cast(tf.shape(var), dtype=tf.int64))
            is_large = max_val > threshold
            tf.cond(is_large,
                    lambda: tf.print(f"WARNING: {var.name} exceeds the threshold of {threshold} at", pos,
                                     "with a value of:", max_val, f"> {threshold}!"),
                    lambda: tf.no_op()
                    )

    @tf.function
    def _print_info(self) -> None:
        pass

    @tf.function
    def _print_splitting_info(self, boundary_inputs: tf.Tensor, boundary_outputs: tf.Tensor,
                              collocation_inputs: tf.Tensor, collocation_outputs: tf.Tensor,
                              inputs: tf.Tensor, outputs: tf.Tensor) -> None:
        tf.print(
            f"Boundary inputs with shape", tf.shape(boundary_inputs), "(of", tf.shape(inputs), "):",
            boundary_inputs
        )
        tf.print(
            f"Boundary outputs with shape", tf.shape(boundary_outputs), "(of", tf.shape(outputs), "):",
            boundary_outputs
        )
        tf.print(
            f"Collocation inputs with shape", tf.shape(collocation_inputs), "(of", tf.shape(inputs), "):",
            collocation_inputs
        )

        tf.print(
            f"Collocation outputs with shape", tf.shape(collocation_outputs), "(of", tf.shape(outputs), "):",
            collocation_inputs
        )

    def _gradients(self, inputs: tf.Tensor) -> None:
        """
        Determines the derivatives of outputs with respect to the inputs in the normalized scale (z-score).
        Args:
            inputs: tf.Tensor of the positional (x, y, and z) data.
        """
        with tf.GradientTape(persistent=True) as tape:
            # Record the computations the inputs are a part of
            tape.watch(inputs)

            with tf.GradientTape(persistent=True) as tape2:
                # Record said computations again (for 2nd derivative)
                tape2.watch(inputs)
                # Update the outputs from the neural network based on its current weights and biases
                self._update_outputs(inputs)

            # Practical function to determine all first derivatives automatically
            _first_deriv: Callable[[str, ], dict[str, tf.Tensor]] = lambda var: {
                # Dict comprehension -> Set gradient to zero if coordinate has no range (thickness of 2D plane)
                #                       or doesn't have a gradient (predictors)
                coord:  tape2.gradient(self._outs[var], inputs)[:, index]
                if index >= 0
                else tf.constant(0.0, dtype=tf.float32)
                for coord,index in self._in_indices.items()
            }

            # Compressible equations
            #self._first_derivs["rho"]  = _first_deriv("rho")

            self._first_derivs["u"]    = _first_deriv("u")
            self._first_derivs["v"]    = _first_deriv("v")
            self._first_derivs["w"]    = _first_deriv("w")
            self._first_derivs["p"]    = _first_deriv("p")
            self._first_derivs["Y_O2"] = _first_deriv("Y_O2")

        # Practical function to determine some second derivatives automatically
        _sec_deriv: Callable[[str, ], dict[str, tf.Tensor]] = lambda var: {
            # Dict comprehension -> Set gradient to zero if coordinate has no range (thickness of 2D plane)
            #                       or doesn't have a gradient (predictors)
            coord: tape.gradient(self._first_derivs[var][coord], inputs)[:, index]
            if index >= 0
            else tf.constant(0.0, dtype=tf.float32)
            for coord,index in self._in_indices.items()
        }

        # Code for compressible equations
        # Practical function to determine all second derivatives automatically
        _sec_deriv_all: Callable[[str, ], dict[str, tf.Tensor]] = lambda var: {
            # Dict comprehension -> Set gradient to zero if coordinate has no range (thickness of 2D plane)
            #                       or doesn't have a gradient (predictors)
            coord2 + coord: tape.gradient(self._first_derivs[var][coord2], inputs)[:, index]
            if index >= 0 and index2 >= 0
            else tf.constant(0.0, dtype=tf.float32)
            for coord2, index2 in self._in_indices.items()
            for coord, index in self._in_indices.items()
        }

        self._sec_derivs["u"]    = _sec_deriv("u")
        self._sec_derivs["v"]    = _sec_deriv("v")
        self._sec_derivs["w"]    = _sec_deriv("w")
        self._sec_derivs["Y_O2"] = _sec_deriv("Y_O2")

    @tf.function
    def _loss_function(self, boundary_inputs: tf.Tensor, boundary_outputs: tf.Tensor,
                       collocation_inputs: tf.Tensor, collocation_outputs: tf.Tensor,
                       inputs: tf.Tensor, outputs: tf.Tensor,
                       inlet_profile_boolean,
                       use_vstats: bool = False) -> tuple[tf.Tensor, dict[str,tf.Tensor]]:

        loss_comps: dict[str, tf.Tensor] = {}
        phys_ins: set[str] = set()
        phys_outs: set[str] = set()
        first_derivs: set[str] = set()
        sec_derivs: set[str] = set()

        # Physical loss
        mse_phys: tf.Tensor = tf.constant(0.0, dtype=tf.float32)

        # Only calculate the data loss without determining any gradients if the ANN is used.
        if not self._is_pinn:
            self._update_outputs(inputs)
            # Only the data loss
            mse_data: tf.Tensor = tf.reduce_mean(
                tf.square(outputs - tf.stack(list(self._outs.values()), axis=1))
            )
            loss_comps["data"] = mse_data
            return mse_data, loss_comps

        # Only calculate the gradients if one of the PDEs requires it; reaction rate doesn't need gradients
        if self._equations["conti"] or self._equations["momentum"] or self._equations["spec_trans"]:
            self._gradients(collocation_inputs)
            if "b" in self._in_indices:
                self._terms_and_scales.update({"GC.Width": 1e-8})

                phys_outs.add("b")

                # Enforce learning the channel width
                width: tf.Tensor = self._calc_bc_loss(
                    boundary_inputs=inputs, boundary_outputs=outputs, var="b", use_vstats=use_vstats
                )
                mse_width: tf.Tensor = self._scale_loss_comp(loss_comp=width, scale=self._terms_and_scales["GC.Width"])
                loss_comps["width"] = mse_width
                mse_phys += mse_width

        if self._equations["conti"]:
            self._terms_and_scales.update({"PDE.Conti": 1e1, "BC.u": 2e-2, "BC.w": 2e-1})

            # Conti Equation
            conti = tf.constant(0, dtype=tf.float32)
            for coord, vel in {"x": "u", "y": "v", "z": "w"}.items():
                if self._in_indices[coord] == -1:
                    continue

                phys_ins.add(coord)
                phys_outs.add(vel)
                first_derivs.add(f"{vel}.{coord}")

                conti += (
                        self._get_var_std(vel, use_vstats) * self._first_derivs[vel][coord] /
                        self._get_var_std(coord, use_vstats)
                )

                # Compressible conti
                """
                conti += (
                        self._denormalize_var("rho", use_vstats) * (
                        self._get_var_std(vel, use_vstats) * self._first_derivs[vel][coord] / 
                        self._get_var_std(coord, use_vstats)
                        ) +
                        self._get_var_std("rho", use_vstats) * (
                                self._denormalize_var(vel, use_vstats) * self._first_derivs["rho"][coord] / 
                                self._get_var_std(coord, use_vstats)
                        )
                )
                """


            mse_conti: tf.Tensor = self._scale_loss_comp(loss_comp=conti, scale=self._terms_and_scales["PDE.Conti"])
            loss_comps["conti"] = mse_conti
            mse_phys += mse_conti

        if self._equations["momentum"]:
            self._terms_and_scales.update({"PDE.Momentum": 1e-2, "BC.u": 2e-2, "BC.w": 2e-1, "BC.p": 1e4})

            phys_outs.add("p")

            # Momentum Equation
            mse_momentum: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
            for mom_coord, mom_vel in {"x": "u", "y": "v", "z": "w"}.items():
                if self._in_indices[mom_coord] == -1:
                    continue

                momentum: tf.Tensor = tf.constant(0, dtype=tf.float32)
                for coord, vel in {"x": "u", "y": "v", "z": "w"}.items():
                    # Convective and diffusive term
                    if self._in_indices[coord] == -1:
                        continue

                    phys_ins.add(mom_coord)
                    phys_outs.add(mom_vel)
                    first_derivs.add(f"{mom_vel}.{coord}")
                    sec_derivs.add(f"{mom_vel}.{coord}")

                    momentum += (
                            (
                                    self._denormalize_var(vel, use_vstats) *
                                    self._first_derivs[mom_vel][coord] / self._get_var_std(coord, use_vstats) -
                                    mom_diff_corr * self._sec_derivs[mom_vel][coord] / reynolds /
                                    tf.square(self._get_var_std(coord, use_vstats))
                            ) * self._get_var_std(mom_vel, use_vstats) *
                            self._get_var_std(mom_coord, use_vstats) / self._get_var_std("p", use_vstats)
                    )

                    # Compressible momentum equation
                    """
                    momentum += (
                        (
                                self._denormalize_var("rho", use_vstats) * self._denormalize_var(vel, use_vstats) * 
                                self._get_var_std(mom_vel, use_vstats) *
                                self._first_derivs[mom_vel][coord] / self._get_var_std(coord, use_vstats) -
                                (
                                        self._get_var_std(mom_vel, use_vstats) * 
                                        self._sec_derivs[mom_vel][coord + coord] /
                                        tf.square(self._get_var_std(coord, use_vstats)) +
                                        tf.constant(1/3, dtype=tf.float32) *
                                        self._get_var_std(vel, use_vstats) / 
                                        (
                                                self._get_var_std(mom_coord, use_vstats) * 
                                                self._get_var_std(coord, use_vstats)
                                        ) *
                                        self._sec_derivs[vel][mom_coord + coord]
                                ) * mom_diff_corr / reynolds
                        ) * self._get_var_std(mom_coord, use_vstats) / self._get_var_std("p", use_vstats)
                    )
                    """

                first_derivs.add(f"p.{mom_coord}")
                # Pressure term
                momentum += (
                        mom_press_corr * self._first_derivs["p"][mom_coord] +
                        mom_press_loss * self._denormalize_var(mom_vel, use_vstats) *
                        self._get_var_std(mom_coord, use_vstats) / self._get_var_std("p", use_vstats)
                )
                mse_momentum += self._scale_loss_comp(loss_comp=momentum, scale=self._terms_and_scales["PDE.Momentum"])

            loss_comps["momentum"] = mse_momentum
            mse_phys += mse_momentum

        if self._equations["spec_trans"]:
            self._terms_and_scales.update({"PDE.Species": 1e3, "BC.Y_O2": 1e-5})

            phys_outs.add("Y_O2")
            
            # Species Transport Equation
            spec_trans: tf.Tensor = tf.constant(0, dtype=tf.float32)
            for coord, vel in {"x": "u", "y": "v", "z": "w"}.items():
                if self._in_indices[coord] == -1:
                    continue

                phys_ins.add(coord)
                phys_outs.add(vel)
                first_derivs.add(f"Y_O2.{coord}")
                sec_derivs.add(f"Y_O2.{coord}")

                spec_trans += (
                        self._denormalize_var(vel, use_vstats) *
                        self._first_derivs["Y_O2"][coord] / self._get_var_std(coord, use_vstats) -
                        spec_diff_corr / peclet *
                        self._sec_derivs["Y_O2"][coord] / tf.square(self._get_var_std(coord, use_vstats))
                )

                # Compressible species transport
                """
                spec_trans += (
                        self._denormalize_var(vel, use_vstats) * self._first_derivs["Y_O2"][coord] / 
                        self._get_var_std(coord, use_vstats) -
                        spec_diff_corr / peclet * (
                                self._sec_derivs["Y_O2"][coord] +
                                self._get_var_std("rho", use_vstats) / 
                                self._denormalize_var("rho", use_vstats) *
                                self._first_derivs["rho"][coord] * self._first_derivs["Y_O2"][coord]
                        ) / tf.square(self._get_var_std(coord, use_vstats))
                )
                """


            mse_spec_trans: tf.Tensor = self._scale_loss_comp(
                loss_comp=spec_trans, scale=self._terms_and_scales["PDE.Species"]
            )
            loss_comps["spec_trans"] = mse_spec_trans
            mse_phys += mse_spec_trans

        if self._equations["react_rate"]:
            self._update_outputs(inputs)

            self._terms_and_scales.update({"AE.Current_Density": 1, "AE.Surface_Overpotential": 1, "CL.Width": 1e-8})
            
            phys_outs.update(["eta", "j"])

            mol_mass: tf.Tensor = tf.pow((  self._denormalize(outputs, "Y_O2", use_vstats) / mol_mass_O2
                                          + self._denormalize(outputs, "Y_H2", use_vstats) / mol_mass_H2
                                          + self._denormalize(outputs, "Y_H2O", use_vstats) / mol_mass_H2O
                                          + self._denormalize(outputs, "Y_N2", use_vstats) / mol_mass_N2),
                                         tf.constant(-1, dtype=tf.float32)
                                         )
            molfrac_O2: tf.Tensor = mol_mass / mol_mass_O2 * self._denormalize(outputs, "Y_O2", use_vstats)
            p_O2: tf.Tensor = molfrac_O2 * self._denormalize(outputs, "p", use_vstats)

            y: tf.Tensor = self._denormalize(inputs, "y", use_vstats)

            if self._single_width is None:
                phys_outs.add("b")

                widths = self._denormalize_var("b", use_vstats)
                # Enforce learning the channel width
                width: tf.Tensor = self._calc_bc_loss(
                    boundary_inputs=inputs, boundary_outputs=outputs, var="b", use_vstats=use_vstats
                )
                mse_width: tf.Tensor = self._scale_loss_comp(loss_comp=width, scale=self._terms_and_scales["CL.Width"])
                loss_comps["width"] = mse_width
                mse_phys += mse_width
            else:
                widths = tf.fill(dims=[tf.shape(inputs)[0]], value=self._single_width)
                widths = tf.cast(widths, dtype=tf.float32)

            resist: tf.Tensor = calc_resist(y=y, channel_widths=widths)

            surf_pot: tf.Tensor = (
                    self._denormalize_var("eta", use_vstats) +
                    self._denormalize(outputs, "U_eq", use_vstats) +
                    resist * mem_area * self._denormalize_var("j", use_vstats)
            )

            # Equation for exchange current density
            exchg0: tf.Tensor = (
                    exchg0_ref * tf.pow((p_O2 / p_ref), gamma)
                    * tf.exp(- act_energy / (gas_const * temp) * (1 - temp / temp_ref)) * 2.1
            )

            model: tf.Tensor = (
                    tf.math.log(exchg0) +
                    gamma_conc * tf.math.log(self._denormalize(outputs, "c_O2", use_vstats)) +
                    alpha_a * faraday * self._denormalize_var("eta", use_vstats) / (gas_const * temp)
            )

            # Equation for current density
            curr_dens: tf.Tensor = (
                    tf.math.log(tf.maximum(self._denormalize_var("j", use_vstats), 1)) - model
            )

            mse_surf_pot: tf.Tensor = self._scale_loss_comp(loss_comp=surf_pot, scale=self._terms_and_scales["AE.Surface_Overpotential"])
            mse_curr_dens: tf.Tensor = self._scale_loss_comp(loss_comp=curr_dens, scale=self._terms_and_scales["AE.Current_Density"])

            loss_comps["surf_pot"] = mse_surf_pot
            loss_comps["curr_dens"] = mse_curr_dens
            mse_phys += mse_surf_pot + mse_curr_dens

        # Boundary conditions
        inlet_profile_boolean = False
        # tf.print("inlet_profile_boolean:", inlet_profile_boolean)
        if inlet_profile_boolean:
            # Consideration of inlet profile
            u_inlet = inputs[:, -4]  # Last 3 columns contain u_inlet, v_inlet, w_inlet in training data
            v_inlet = inputs[:, -3]
            w_inlet = inputs[:, -2]
            p_inlet = inputs[:, -1]

            inlet_mask = tf.math.equal(inputs[:, 0], tf.reduce_min(inputs[:, 0]))  # Mask for x = x_min
            u_inlet_pred = tf.boolean_mask(self._outs["u"], inlet_mask)
            v_inlet_pred = tf.boolean_mask(self._outs["v"], inlet_mask)
            w_inlet_pred = tf.boolean_mask(self._outs["w"], inlet_mask)
            p_inlet_pred = tf.boolean_mask(self._outs["p"], inlet_mask)

            mse_inlet = tf.reduce_mean(
                tf.square(u_inlet_pred - u_inlet) +
                tf.square(v_inlet_pred - v_inlet) +
                tf.square(w_inlet_pred - w_inlet) +
                tf.square(p_inlet_pred - p_inlet)
            )
        else:
            mse_inlet: tf.Tensor = tf.constant(0, dtype=tf.float32)

        # Add boundary losses
        for var, scale in self._terms_and_scales.items():
            var_type, var_name = var.split(".")
            if var_type != "BC":
                continue
            
            boundary: tf.Tensor = self._calc_bc_loss(
                boundary_inputs=boundary_inputs, boundary_outputs=boundary_outputs, var=var_name, use_vstats=use_vstats
            )
            mse_boundary: tf.Tensor = self._scale_loss_comp(
                loss_comp=boundary, scale=scale
            )
            loss_comps[f"bc_{var_name}"] = mse_boundary
            mse_phys += mse_boundary

        # Print inputs as well as outputs and their derivatives
        if self._debug:
            if phys_ins:
                tf.print(10 * "=")
                tf.print("Inputs:")
            for coord in phys_ins:
                tf.print(f"{coord}_std: ", self._get_var_std(coord, use_vstats))

            if phys_outs:
                tf.print(10 * "=")
                tf.print("Outputs:")
            for phys_out in phys_outs:
                tf.print(f"{phys_out}: ", self._denormalize_var(phys_out, use_vstats))
                tf.print(f"{phys_out}_std: ", self._get_var_std(phys_out, use_vstats))

            if first_derivs:
                tf.print(10 * "=")
                tf.print("First Derivatives:")
            for first_deriv in first_derivs:
                first_deriv_out, first_deriv_in = first_deriv.split(".")
                tf.print(f"d{first_deriv_out}/d{first_deriv_in}: ", self._first_derivs[first_deriv_out][first_deriv_in])

            if sec_derivs:
                tf.print(10 * "=")
                tf.print("Second Derivatives:")
            for sec_deriv in sec_derivs:
                sec_deriv_out, sec_deriv_in = sec_deriv.split(".")
                tf.print(f"d^2{sec_deriv_out}/d{sec_deriv_in}^2: ", self._sec_derivs[sec_deriv_out][sec_deriv_in])

            tf.print(10 * "=")
            tf.print("Losses:")
            for loss_name, loss_val in loss_comps.items():
                tf.print(f"{loss_name}: ", loss_val)
            tf.print(10 * "=")
        
        # Total loss
        loss: tf.Tensor = mse_phys #+ mse_data # + mse_inlet
        return loss, loss_comps

    @tf.function
    def get_train_loss(self, collocation_inputs: tf.Tensor, boundary_inputs: tf.Tensor,
                 collocation_outputs: tf.Tensor, boundary_outputs: tf.Tensor,
                 inputs: tf.Tensor, outputs: tf.Tensor,
                 inlet_profile_boolean) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:

        return self._loss_function(
            collocation_inputs=collocation_inputs,
            boundary_inputs=boundary_inputs,
            collocation_outputs=collocation_outputs,
            boundary_outputs=boundary_outputs,
            inlet_profile_boolean=inlet_profile_boolean,
            inputs=inputs,
            outputs=outputs,
            use_vstats=False
        )

    @tf.function
    def get_val_loss(self, collocation_inputs: tf.Tensor, boundary_inputs: tf.Tensor,
                   collocation_outputs: tf.Tensor, boundary_outputs: tf.Tensor,
                   inputs: tf.Tensor, outputs: tf.Tensor,
                   inlet_profile_boolean) -> tuple[tf.Tensor,dict[str,tf.Tensor]]:

        loss, loss_comps = self._loss_function(
            collocation_inputs=collocation_inputs,
            boundary_inputs=boundary_inputs,
            collocation_outputs=collocation_outputs,
            boundary_outputs=boundary_outputs,
            inlet_profile_boolean=inlet_profile_boolean,
            inputs=inputs,
            outputs=outputs,
            use_vstats=True
        )
        
        return loss, loss_comps

    def get_terms_and_scales(self) -> dict[str,float]:
        return self._terms_and_scales
    
    def get_trainable_variables(self) -> list[tf.Variable]:
        return self._pinn.trainable_variables

    def get_pinn(self) -> PINN:
        return self._pinn