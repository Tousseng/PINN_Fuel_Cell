from __future__ import annotations

from training.training import Training
from training.optimizer.optimizer import Optimizer
from utilities import split_collocation_and_boundary

import time
from typing_extensions import override

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

class LBFGS(Optimizer):

    @override
    def _run(self, train_data: dict[str,np.ndarray], val_data: dict[str,np.ndarray] | None,
             start_iteration: int, inlet_profile: np.ndarray) -> None:
        start_time: float = time.time()
        print(f"Running the L-BFGS-Optimizer.")
        print(20 * "-")
        if self._iters > 0:
            bound_inputs, bound_outputs, coll_inputs, coll_outputs, inputs, outputs = split_collocation_and_boundary(
                inputs=tf.convert_to_tensor(train_data["in"], dtype=tf.float32),
                outputs=tf.convert_to_tensor(train_data["out"], dtype=tf.float32),
                in_indices=self._training.get_in_indices(),
                epsilon=self._epsilon
            )

            # If no data is available, create empty tensors for the validation data
            if val_data is None:
                self._val_interval = None
                val_bound_inputs = tf.constant([], dtype=tf.float32)
                val_bound_outputs = val_bound_inputs
                val_coll_inputs = val_bound_inputs
                val_coll_outputs = val_bound_inputs
                val_inputs = val_bound_inputs
                val_outputs = val_bound_inputs
            else:
                val_bound_inputs, val_bound_outputs, val_coll_inputs, val_coll_outputs, val_inputs, val_outputs = (
                    split_collocation_and_boundary(
                        inputs=tf.convert_to_tensor(val_data["in"], dtype=tf.float32),
                        outputs=tf.convert_to_tensor(val_data["out"], dtype=tf.float32),
                        in_indices=self._training.get_in_indices(),
                        epsilon=self._epsilon
                    )
                )

            self._loss_history, self._loss_comps, self._val_loss_history, self._val_loss_comps = (
                self._lbfgs_optimization_step(
                    collocation_inputs=coll_inputs,
                    boundary_inputs=bound_inputs,
                    collocation_outputs=coll_outputs,
                    boundary_outputs=bound_outputs,
                    inputs=inputs,
                    outputs=outputs,
                    val_collocation_inputs=val_coll_inputs,
                    val_boundary_inputs=val_bound_inputs,
                    val_collocation_outputs=val_coll_outputs,
                    val_boundary_outputs=val_bound_outputs,
                    val_inputs=val_inputs,
                    val_outputs=val_outputs,
                    inlet_profile_boolean=False,
                    max_iter=self._iters,
                    target_val=self._target_val,
                    val_interval=self._val_interval,
                    start_iteration=start_iteration
                )
            )

        self._opt_time = time.time() - start_time
        self._iters = len(self._loss_history)
        self._val_iters = len(self._val_loss_history)
        if self._print_comps:
            self._print_loss_comps()
        print("Loss history:", self._loss_history)

    def _get_weights_and_gradients(self):
        weights = tf.concat([tf.reshape(v, [-1]) for v in self._training.get_trainable_variables()], axis=0)

        def set_weights(weights):
            weights = tf.cast(weights, tf.float32)  # Convert weights to float32
            start = 0
            for var in self._training.get_trainable_variables():
                shape = tf.shape(var)
                size = tf.reduce_prod(shape)
                var.assign(tf.reshape(weights[start:start + size], shape))
                start += size

        return weights, set_weights

    def _lbfgs_optimization_step(self, collocation_inputs: tf.Tensor, boundary_inputs: tf.Tensor,
                                collocation_outputs: tf.Tensor, boundary_outputs: tf.Tensor,
                                inputs: tf.Tensor, outputs: tf.Tensor,
                                val_collocation_inputs: tf.Tensor, val_boundary_inputs: tf.Tensor,
                                val_collocation_outputs: tf.Tensor, val_boundary_outputs: tf.Tensor,
                                val_inputs: tf.Tensor, val_outputs: tf.Tensor,
                                inlet_profile_boolean: bool, max_iter: int, target_val: float | None,
                                val_interval: int | None, start_iteration: int):
        weights, set_weights = self._get_weights_and_gradients()

        def loss_and_gradients(weights):
            set_weights(weights)
            with tf.GradientTape() as tape:
                loss, loss_comps = self._training.get_train_loss(
                    collocation_inputs=collocation_inputs,
                    boundary_inputs=boundary_inputs,
                    collocation_outputs=collocation_outputs,
                    boundary_outputs=boundary_outputs,
                    inlet_profile_boolean=inlet_profile_boolean,
                    inputs=inputs,
                    outputs=outputs
                )
            gradients = tape.gradient(loss, self._training.get_trainable_variables())

            self._training.find_weight_over_threshold(grads=gradients)

            gradients_flat = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
            loss_history.append(loss.numpy())
            if not loss_comps_complete:
                for loss_name, loss_val in loss_comps.items():
                    loss_comps_complete[loss_name] = [loss_val.numpy()]
            else:
                for loss_name, loss_val in loss_comps.items():
                    loss_comps_complete[loss_name].append(loss_val.numpy())
            return loss.numpy(), gradients_flat.numpy()

        def callback(weights):
            iteration = len(loss_history)
            if loss_history:
                current_loss = loss_history[-1]
                print(f"Iteration: {iteration}, Loss: {current_loss}")
                iteration_losses.append(current_loss)

            if loss_comps_complete:
                if not iteration_loss_comps:
                    for loss_name, loss_val in loss_comps_complete.items():
                        iteration_loss_comps[loss_name] = [loss_val[-1]]
                else:
                    for loss_name, loss_val in loss_comps_complete.items():
                        iteration_loss_comps[loss_name].append(loss_val[-1])

            if val_interval is not None and iteration % val_interval == 0:
                val_loss, val_loss_comps = self._training.get_val_loss(
                    collocation_inputs=val_collocation_inputs,
                    boundary_inputs=val_boundary_inputs,
                    collocation_outputs=val_collocation_outputs,
                    boundary_outputs=val_boundary_outputs,
                    inputs=val_inputs,
                    outputs=val_outputs,
                    inlet_profile_boolean=inlet_profile_boolean,
                )
                val_loss_history.append(val_loss.numpy())
                print(f"VALIDATION CYCLE - Iteration: {iteration}, Loss: {val_loss.numpy()}")
                if not val_iteration_loss_comps:
                    for loss_name, loss_val in val_loss_comps.items():
                        loss_comps_complete[loss_name] = [loss_val.numpy()]
                else:
                    for loss_name, loss_val in val_loss_comps.items():
                        loss_comps_complete[loss_name].append(loss_val.numpy())

            set_weights(weights)

            if not target_val is None and iteration_losses[-1] < target_val:
                raise StopIteration
            self._save_model_during_training(curr_iteration=start_iteration+len(iteration_losses))
            return iteration_losses, iteration_loss_comps

        iteration_losses = []
        loss_history = []
        val_loss_history: list[float] = []
        loss_comps_complete: dict[str,list[float]] = {}
        iteration_loss_comps: dict[str,list[float]] = {}
        val_iteration_loss_comps: dict[str,list[float]] = {}
        x0 = weights.numpy()

        try:
            result = minimize(
                fun = loss_and_gradients,
                x0 = x0,
                method = 'L-BFGS-B',
                jac = True,
                callback=callback,
                options = {
                    'disp' : True,
                    'maxiter': max_iter,
                    'maxfun' : max_iter*1.5,
                    'maxcor' : 50,
                    'maxls': 10,
                    'gtol': 1.0 * np.finfo(float).eps,
                    'ftol' : 1.0 * np.finfo(float).eps
                }
            )
        except StopIteration as stop:
            print(stop)
            print(f"Loss target was reached: Loss = {iteration_losses[-1]} < {target_val}")

        # ATTENTION: 'loss_history' and 'loss_comps_complete' are not returned even though they are available
        return iteration_losses, iteration_loss_comps, val_loss_history, val_iteration_loss_comps

    @override
    def _iteration_summary(self, *args) -> str:
        pass