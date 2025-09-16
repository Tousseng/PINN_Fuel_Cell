from __future__ import annotations

from training.training import Training
from training.optimizer.optimizer import Optimizer
from utilities import convert_time, split_collocation_and_boundary

import time
from typing import Callable
from typing_extensions import override

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam as Adam_tf

class Adam(Optimizer):
    def __init__(self, training: Training, metrics: dict[str,int|float|None],
                 model_path: str, save_iters: list[int] | None = None):
        super().__init__(
            training=training, metrics=metrics, model_path=model_path, save_iters=save_iters
        )
        self._optimizer: Adam_tf = Adam_tf(learning_rate=metrics["learning_rate"])
        self._batch_size: int = metrics["batch_size"]
        self._avg_loss_history: list[float] = []
        self._avg_val_loss_history: list[float] = []

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def set_learning_rate(self, learning_rate: float) -> None:
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def _save_loss_comps(loss_comps_attribute: dict[str,list[float]], loss_comps: dict[str,float]) -> None:
        if not loss_comps_attribute:
                for loss_name, loss_val in loss_comps.items():
                    loss_comps_attribute[loss_name] = [loss_val]
        else:
            for loss_name, loss_val in loss_comps.items():
                loss_comps_attribute[loss_name].append(loss_val)

    @staticmethod
    def _add_epoch_loss_comps(loss_comps: dict[str,tf.Tensor], loss_comps_epoch: dict[str,float]) -> dict[str,float]:
        if not loss_comps_epoch:
                for loss_name, loss_val in loss_comps.items():
                    loss_comps_epoch[loss_name] = loss_val.numpy()
        else:
            for loss_name, loss_val in loss_comps.items():
                loss_comps_epoch[loss_name] += loss_val.numpy()
        return loss_comps_epoch

    def _avg_loss_comps(self, loss_comps_epoch: dict[str,float], data_size: int) -> dict[str,float]:
        for loss_name, loss_val in loss_comps_epoch.items():
            loss_comps_epoch[loss_name] = loss_val / (data_size / self._batch_size)
        return loss_comps_epoch

    @override
    def _run(self, train_data: dict[str,np.ndarray], val_data: dict[str,np.ndarray] | None,
             start_iteration: int, inlet_profile: np.ndarray) -> None:
        print(f"Running the Adam-Optimizer.")
        print(20 * "-")
        bound_inputs, bound_outputs, coll_inputs, coll_outputs, inputs, outputs = split_collocation_and_boundary(
            inputs=tf.convert_to_tensor(train_data["in"], dtype=tf.float32),
            outputs=tf.convert_to_tensor(train_data["out"], dtype=tf.float32),
            in_indices=self._training.get_in_indices(),
            epsilon=self._epsilon
        )
        val_bound_inputs, val_bound_outputs, val_coll_inputs, val_coll_outputs, val_inputs, val_outputs = (
            split_collocation_and_boundary(
                inputs=tf.convert_to_tensor(val_data["in"], dtype=tf.float32),
                outputs=tf.convert_to_tensor(val_data["out"], dtype=tf.float32),
                in_indices=self._training.get_in_indices(),
                epsilon=self._epsilon
            )
        )
        for epoch in range(start_iteration, start_iteration + self._iters):
            time_start: float = time.time()
            epoch_loss: float = 0.0
            val_epoch_loss: float = 0.0
            loss_comps_epoch: dict[str,float] = {}
            val_loss_comps_epoch: dict[str, float] = {}

            if self._val_interval is not None and epoch % self._val_interval == 0:
                self._epoch_iteration(
                    data=val_data["in"],
                    func=self._training.get_val_loss,
                    coll_inputs=val_coll_inputs,
                    bound_inputs=val_bound_inputs,
                    coll_outputs=val_coll_outputs,
                    bound_outputs=val_bound_outputs,
                    inputs=val_inputs,
                    outputs=val_outputs,
                    epoch=epoch,
                    epoch_loss=val_epoch_loss,
                    time_start=time_start,
                    avg_loss_history=self._avg_val_loss_history,
                    iter_time_history=self._val_iter_time_history,
                    loss_comps_epoch=val_loss_comps_epoch,
                    loss_comps_attribute=self._val_loss_comps,
                    msg="VALIDATION CYCLE - "
                )

            self._epoch_iteration(
                data=train_data["in"],
                func=self._train_step,
                coll_inputs=coll_inputs,
                bound_inputs=bound_inputs,
                coll_outputs=coll_outputs,
                bound_outputs=bound_outputs,
                inputs=inputs,
                outputs=outputs,
                epoch=epoch,
                epoch_loss=epoch_loss,
                time_start=time_start,
                avg_loss_history=self._avg_loss_history,
                iter_time_history=self._iter_time_history,
                loss_comps_epoch=loss_comps_epoch,
                loss_comps_attribute=self._loss_comps
            )

            if self._print_comps:
                self._print_loss_comps()
            # Print epoch summary
            print(self._iteration_summary(epoch))
            print(20 * "-")
            if self._target_val is not None and self._avg_loss_history[-1] < self._target_val:
                break
            self._save_model_during_training(curr_iteration=len(self._avg_loss_history))

        self._iters = len(self._avg_loss_history)
        self._val_iters = len(self._avg_val_loss_history)

    def _epoch_iteration(self, data: np.ndarray, func: Callable,
                         coll_inputs: tf.Tensor, bound_inputs: tf.Tensor, coll_outputs: tf.Tensor,
                         bound_outputs: tf.Tensor, inputs: tf.Tensor, outputs: tf.Tensor,
                         epoch: int, epoch_loss: float, time_start: float,
                         avg_loss_history: list[float], iter_time_history: list[float],
                         loss_comps_epoch: dict[str,float], loss_comps_attribute: dict[str,list[float]],
                         msg: str = "") -> None:
        # Iterate over the dataset in batches
        for i in range(0, len(data), self._batch_size):
            # Perform a training step and calculate the loss
            loss, loss_comps = func(
                collocation_inputs=coll_inputs[i:i + self._batch_size, :],
                boundary_inputs=bound_inputs,
                collocation_outputs=coll_outputs[i:i + self._batch_size, :],
                boundary_outputs=bound_outputs,
                inputs=inputs[i:i + self._batch_size, :],
                outputs=outputs[i:i + self._batch_size, :],
                inlet_profile_boolean=False
            )  ### CURRENTLY *inlet_profile_boolean* IS NOT USED.
            loss_comps_epoch = self._add_epoch_loss_comps(loss_comps, loss_comps_epoch)
            print(f"{msg}Epoch/Iteration: {epoch + 1}/{i}; Loss: {loss}")
            # Accumulate the loss for the current epoch
            epoch_loss += loss
            self._loss_history.append(loss.numpy())

        # Calculate average loss for the epoch and store it
        avg_loss_history.append((epoch_loss / (len(data) / self._batch_size)).numpy())
        iter_time_history.append(time.time() - time_start)
        loss_comps_epoch = self._avg_loss_comps(loss_comps_epoch, len(data))
        self._save_loss_comps(loss_comps_attribute, loss_comps_epoch)

    @tf.function
    def _train_step(self, collocation_inputs: tf.Tensor, boundary_inputs: tf.Tensor,
                   collocation_outputs: tf.Tensor, boundary_outputs: tf.Tensor,
                   inputs: tf.Tensor, outputs: tf.Tensor,
                   inlet_profile_boolean) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
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
        grads = tape.gradient(loss, self._training.get_trainable_variables())

        self._training.find_weight_over_threshold(grads=grads)

        self._optimizer.apply_gradients(zip(grads, self._training.get_trainable_variables()))
        return loss, loss_comps

    @override
    def _iteration_summary(self, *args) -> str:
        return (
            f"Epoch: {args[0] + 1} / {self._iters}\n"
            f"Average Loss: {self._avg_loss_history[-1]}\n"
            f"Process Time: {convert_time(self._iter_time_history[-1])}"
        )

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_learning_rate(self) -> float:
        return self._optimizer.learning_rate.numpy()

    def get_avg_loss_history(self, train_info: bool) -> list[float]:
        return self._avg_loss_history if train_info else self._avg_val_loss_history