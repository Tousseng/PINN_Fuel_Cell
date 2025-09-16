from __future__ import annotations

from keras.utils.version_utils import training

from training.pinn import PINN
from training.training import Training

import os
import time
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

class Optimizer(ABC):
    def __init__(self, training: Training, metrics: dict[str,int|float|None], model_path: str,
                 save_iters: list[int] | None = None):
        self._training: Training = training
        self._iters: int = metrics["iterations"]
        self._save_iters: list[int] | None = save_iters
        self._val_iters: int = 0
        self._target_val: float | None = metrics["target_val"]
        self._val_interval: int | None = metrics["val_interval"]
        self._print_comps: bool = False
        self._epsilon: float = 1
        self._loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._iter_time_history: list[float] = []
        self._val_iter_time_history: list[float] = []
        self._loss_comps: dict[str, list[float]] = {}
        self._val_loss_comps: dict[str,list[float]] = {}
        self._opt_time: float = 0.0
        self._terms_and_scales: dict[str,float] = training.get_terms_and_scales()
        self._model_path: str = model_path

    def set_iters(self, iterations: int) -> None:
        self._iters = iterations

    def run(self, device: str, train_data: dict[str,np.ndarray], val_data: dict[str,np.ndarray] | None,
            start_iteration: int = 0, inlet_profile: np.ndarray = None) -> PINN:
        start_time: float = time.time()
        with tf.device(device):
            self._run(
                train_data=train_data, val_data=val_data,
                start_iteration=start_iteration, inlet_profile=inlet_profile
            )
        self._opt_time = time.time() - start_time
        pinn: PINN = self._training.get_pinn()
        del self._training
        return pinn


    def _print_loss_comps(self) -> None:
        print("### Loss components ###")
        for loss_name, loss_val in self._loss_comps.items():
            print(f"{loss_name}: {loss_val}")
        print("### Loss components ###")

    @abstractmethod
    def _run(self, train_data: dict[str,np.ndarray], val_data: dict[str,np.ndarray] | None,
             start_iteration: int, inlet_profile: np.ndarray) -> None:
        pass

    @abstractmethod
    def _iteration_summary(self, *args) -> str:
        pass

    def get_loss_history(self, train_info: bool) -> list[float]:
        return self._loss_history if train_info else self._val_loss_history

    def get_iter_time_history(self, train_info: bool) -> list[float]:
        return self._iter_time_history if train_info else self._val_iter_time_history

    def get_iters(self, train_info) -> int:
        return self._iters if train_info else self._val_iters

    def get_opt_time(self) -> float:
        return self._opt_time

    def get_loss_comps(self, train_info: bool) -> dict[str,list[float]]:
        return self._loss_comps if train_info else self._val_loss_comps

    def get_val_interval(self) -> int | None:
        return self._val_interval

    def get_target_val(self) -> float | None:
        return self._target_val

    def get_terms_and_scales(self) -> dict[str,float]:
        return self._terms_and_scales

    def _save_model_during_training(self, curr_iteration: int) -> None:
        if self._save_iters is None:
            return None

        if curr_iteration in self._save_iters:
            pinn: PINN = self._training.get_pinn()
            save_path: str = os.path.join(
                self._model_path, f"model_at_{curr_iteration}_iterations"
            )
            print(10 * "+")
            print(f"Saving model at {curr_iteration} iterations.")
            print(10 * "+")
            os.makedirs(save_path, exist_ok=True)
            pinn.save(save_path)