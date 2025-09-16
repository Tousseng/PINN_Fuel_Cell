from __future__ import annotations

from training.pinn import PINN
from data_handling.data import Data
from training.training import Training
from training.optimizer import Adam, LBFGS
from log_handling import Logger, dict_to_str
from postprocessing import plot_loss, plot_variation_loss
from utilities import timer, convert_time, get_datetime, convert_var_name_rev

import os
import gc
import shutil
from tkinter import filedialog

from tensorflow.keras.backend import clear_session

class Setup:
    """
    Encapsulates the user defined input.
    """

    def __init__(self, data: Data,
                 metrics: dict[str,dict[str,int|float|None]], model_dir: str, predictors: list[str],
                 equations: dict[str,bool], single_width: float | None = None, msg: str = "",
                 save_iters: list[int] | None = None, debug: bool = False):
        self._data: Data = data
        self._val_data: Data | None = None
        self._adam: Adam | None = None
        self._lbfgs: LBFGS | None = None
        self._data_metrics: dict[str,float] = metrics["data_metrics"]
        self._pinn_metrics: dict[str,int] = metrics["pinn_metrics"]
        self._adam_metrics: dict[str,int|float] = metrics["adam_metrics"]
        self._lbfgs_metrics: dict[str,int|float] = metrics["lbfgs_metrics"]
        self._predictors: list[str] = predictors
        self._save_iters: list[int] | None = save_iters
        self._loss_history: dict[str,list[float]] = {}
        self._val_loss_history: dict[str,list[float]] = {}
        self._loss_comps_history: dict[str,list[float]] = {}
        self._val_loss_comps_history: dict[str,list[float]] = {}
        self._iter_time_history: dict[str,dict[str,float]] = {}
        self._val_iter_time_history: dict[str,dict[str,float]] = {}
        self._varying_param: str = ""
        self._model_dir: str = model_dir
        self._model_path: str = ""
        self._is_first: bool = True
        self._equations: dict[str,bool] = equations
        self._single_width: float | None = single_width
        self._debug: bool = debug
        self._msg: str = msg

    def set_var(self, varying_param: str) -> None:
        self._varying_param = varying_param

    def reset_history(self) -> None:
        self._loss_history = {}
        self._val_loss_history = {}
        self._loss_comps_history = {}
        self._val_loss_comps_history = {}
        self._iter_time_history = {}
        self._val_iter_time_history = {}
        self._is_first = True
        self._model_dir = "../trained_model_directory"
        self._model_path = ""

    def reset_training(self) -> None:
        del self._adam
        del self._lbfgs
        gc.collect()
        clear_session()

    def _add_readme(self) -> None:
        if self._msg == "":
            return None

        file_name: str = os.path.join(self._model_path, "README.txt")
        with open(file_name, "w") as f:
            f.write(self._msg)

    def _setup_dirs(self, val: int | float) -> None:
        var_str: str = f"_var_{self._varying_param}" if self._varying_param != "" else ""
        # Setup for saving the model
        model_name: str = os.path.basename(self._data.get_data_file()).split(".")[0]
        if self._varying_param != "":
            if self._is_first:
                model_name = "CL_model" if "CL" in model_name else "GC_model"
                self._model_dir: str = os.path.join(self._model_dir, f"{model_name}@{get_datetime()}{var_str}")
                os.makedirs(self._model_dir, exist_ok=True)
                self._is_first = False
            varying_param_model_name: str = f"{self._varying_param}_{val}"
            if self._varying_param == "model":
                varying_param_model_name = f"{'PINN' if self._is_pinn() else 'ANN'}_" + varying_param_model_name
            self._model_path: str = os.path.join(
                self._model_dir, varying_param_model_name
            )
        else:
            self._model_path: str = os.path.join(self._model_dir, f"{model_name}@{get_datetime()}")
        os.makedirs(self._model_path, exist_ok=True)
        shutil.copyfile(
            src=self._data.get_data_file(), dst=f"{os.path.join(self._model_path, model_name)}.csv"
        )
        if self._val_data is not None:
            val_model_name = os.path.basename(self._val_data.get_data_file()).split(".")[0]
            shutil.copyfile(
                src=self._val_data.get_data_file(), dst=f"{os.path.join(self._model_path, val_model_name)}.csv"
            )
        self._add_readme()

    def _print_inputs_outputs(self) -> None:
        print(5 * "*")
        print(f"The following variables will be retrieved from the data...")
        print(f"...inputs: {self._data.get_input_vars()}")
        print(f"...outputs: {self._data.get_output_vars()}")
        print(5 * "*")

    def load_train_data(self, file_path: str) -> None:
        print(f"Loading data from {file_path}.")
        self._data.add_input_vars(self._predictors)
        self._data.add_output_vars(self._predictors)
        self._print_inputs_outputs()
        _, load_time = self._load_data(data=self._data, file_path=file_path)
        print(f"Done. Loading took {convert_time(load_time)}")
        print(20 * "-")

    def load_val_data(self) -> None:
        # Do not use validation data if both validation intervals are zero.
        if self._adam_metrics["val_interval"] is None and self._lbfgs_metrics["val_interval"] is None:
            return None
        # Create the data object for the validation data based on the same coordinate frame
        # and output variables as the training data (same reference)
        self._val_data = Data(
            coord_frame=self._data.get_coords(),
            outputs=self._data.get_output_vars()
        )
        file_path: str = filedialog.askopenfilename(title="Choose a CSV file for validating")
        print(f"Loading data from {file_path}.")
        # Channel widths will already be set in outputs during self._val_data object creation
        self._val_data.add_input_vars(self._predictors)
        self._print_inputs_outputs()
        _, load_time = self._load_data(data=self._val_data, file_path=file_path)
        print(f"Done. Loading took {convert_time(load_time)}")
        print(20 * "-")

    def setup_pinn(self, val: int | float) -> None:
        print(f"Creating optimizers")
        _, create_optimizers_time = self._create_optimizers(val)
        print(f"Done. Creating training took {convert_time(create_optimizers_time)}")
        print(20 * "-")

    def train_pinn(self, val: int | float) -> None:
        pinn: PINN = self._run_adam(val)
        print(f"Done. Running Adam took {convert_time(self._adam.get_opt_time())}")
        print(20 * "-")
        self._run_lbfgs(val)
        print(f"Done. Running L-BFGS took {convert_time(self._lbfgs.get_opt_time())}")
        print(20 * "-")

        if self._save_iters is None or len(self._save_iters) == 0:
            self._setup_dirs(val)
        pinn.save(self._model_path)
        self._log_results(val, pinn)

    def _is_pinn(self) -> bool:
        for use_eq in list(self._equations.values()):
            if use_eq:
                return True
        return False

    @timer
    def _load_data(self, data: Data, file_path: str) -> None:
        """
        Calls *load_data()* of *data* object.\n
        **NOT DESIGNED TO BE USED ON ITS OWN**.
        Args:
            file_path: Path to the file to load.
        """
        data.load_data(file_path=file_path,vel_lim=self._data_metrics["vel_lim"],
                       tol=self._data_metrics["tol"],
                       slice_pos=self._data_metrics["slice_pos"], shuffle=True)
        # Channel width is a predictor and can be found in the outputs as well as the inputs
        if convert_var_name_rev("b") in data.get_output_vars():
            self._single_width = None
        else:
            # Channel width is already supplied by the csv-file but is no input nor output
            if convert_var_name_rev("b") in list(data.get_data_slice().columns):
                # Retrieve the first value inside the channel widths column
                norm_stats_width: dict[str,float] = data.get_norm_stats()[convert_var_name_rev("b")]
                width_mean: float = norm_stats_width["mean"]
                width_std: float = norm_stats_width["std"]
                if width_std > 1e-9:
                    raise ValueError(
                        f"A standard deviation of {width_std} is too high "
                        f"for a single value of {convert_var_name_rev('b')}. "
                        f"Check if you maybe didn't set the 'predictors' option."
                    )
                width_norm: float = data.get_data_slice()[convert_var_name_rev("b")][0]
                self._single_width = width_mean + width_std * width_norm
            else:
                # Enforce that channel width is set if reaction rate is used
                if self._equations["react_rate"] and self._single_width is None:
                    raise ValueError(
                        f"'single_width' can't be {self._single_width} and needs to be supplied "
                        f"since it can't be found in the data itself."
                    )

    @timer
    def _create_pinn(self, val: int | float) -> PINN:
        """
        Automatically creates a PINN consistent with the *setup* object.\n
        **NOT DESIGNED TO BE USED ON ITS OWN**.
        """
        pinn: PINN = PINN(
            num_layers=self._pinn_metrics["num_layers"],
            num_neurons=self._pinn_metrics["num_neurons"],
            input_dim=self._data.get_num_inputs(),
            output_dim=self._data.get_num_outputs()
        )

        if self._varying_param == "layers":
            pinn.set_num_layers(val)

        if self._varying_param == "neurons":
            pinn.set_num_neurons(val)
        return pinn

    @timer
    def _create_training(self, val: int | float) -> Training:
        """
        Automatically creates a training object consistent with the *setup* object.\n
        **NOT DESIGNED TO BE USED ON ITS OWN**.
        """
        print(f"Creating PINN")
        pinn, create_pinn_time = self._create_pinn(val)
        print(f"Done. Creating PINN took {convert_time(create_pinn_time)}")
        print(20 * "-")

        return Training(
            pinn=pinn,
            in_indices=self._data.get_in_indices(),
            out_indices=self._data.get_out_indices(),
            equations=self._equations,
            norm_stats=self._data.get_norm_stats(),
            val_norm_stats=None if self._val_data is None else self._val_data.get_norm_stats(),
            is_pinn=self._is_pinn(),
            single_width=self._single_width,
            debug=self._debug
        )

    @timer
    def _create_optimizers(self, val: int | float) -> None:
        print(f"Creating training")
        training, create_training_time = self._create_training(val)
        print(f"Done. Creating training took {convert_time(create_training_time)}")
        print(20 * "-")
        if self._save_iters is not None and len(self._save_iters) > 0:
            self._setup_dirs(val)
            interim_model_path: str = os.path.join(self._model_path, "interim_models")
            os.makedirs(interim_model_path, exist_ok=True)
            adam_index: int = max(
                (i for i, v in enumerate(self._save_iters) if v <= self._adam_metrics["iterations"]), default=0
            )
            adam_save_iters: list[int] | None = self._save_iters[:adam_index+1]
            lbfgs_save_iters: list[int] | None = self._save_iters[adam_index+1:]
        else:
            interim_model_path: str = self._model_path
            adam_save_iters: list[int] | None = None
            lbfgs_save_iters: list[int] | None = None
        self._adam = Adam(
            training=training, metrics=self._adam_metrics, model_path=interim_model_path, save_iters=adam_save_iters
        )
        self._lbfgs = LBFGS(
            training=training, metrics=self._lbfgs_metrics, model_path=interim_model_path, save_iters=lbfgs_save_iters
        )
        # If varying parameters are used, the hyperparameters of the optimizers are changed after they are initialized
        # to keep the base values in the form of the 'self._adam_metrics' and 'self._lbfgs_metrics' intact
        if self._varying_param == "epochs":
            self._adam.set_iters(val)

        if self._varying_param == "batch_size":
            self._adam.set_batch_size(val)

        if self._varying_param == "learning_rate":
            self._adam.set_learning_rate(val)

        if self._varying_param == "iterations":
            self._lbfgs.set_iters(val)

    #@timer
    def _run_adam(self, val: int | float) -> PINN:
        pinn: PINN = self._adam.run(
            device="/device:GPU:0",
            train_data=self._data.get_train_data() if self._val_data is None
            else self._data.get_train_data(val_fraction=0),
            # self._val_data != None will be set at Runtime
            val_data=self._data.get_val_data() if self._val_data is None
            else self._val_data.get_train_data(val_fraction=0), # type: ignore
        )
        self._save_adam_metrics(val)
        return pinn

    def _save_adam_metrics(self, val: int | float) -> None:
        self._loss_history[str(val)] = self._adam.get_avg_loss_history(train_info=True)
        self._val_loss_history[str(val)] = self._adam.get_avg_loss_history(train_info=False)
        # First entry of key 'Adam' in dict
        if self._iter_time_history.get("Adam") is None:
            self._iter_time_history["Adam"] = {str(val): self._adam.get_opt_time()}
        # Update underlying dict to key 'Adam'
        else:
            self._iter_time_history["Adam"].update({str(val): self._adam.get_opt_time()})
        self._copy_loss_comps(val=val, train_info=True)
        self._copy_loss_comps(val=val, train_info=False)

    def _copy_loss_comps(self, val: int | float, train_info: bool) -> None:
        loss_comps: dict[str, list[float]] = self._adam.get_loss_comps(train_info=train_info)
        if train_info:
            self._loss_comps_history["total"] = self._loss_history[str(val)]
        else:
            self._val_loss_comps_history["total"] = self._val_loss_history[str(val)]
        for loss_name, loss_vals in loss_comps.items():
            if train_info:
                self._loss_comps_history[loss_name] = loss_vals
            else:
                self._val_loss_comps_history[loss_name] = loss_vals

    #@timer
    def _run_lbfgs(self, val: int | float) -> PINN:
        pinn: PINN = self._lbfgs.run(
            device="/device:CPU:0",
            train_data={"in": self._data.get_inputs(),"out": self._data.get_outputs()},
            val_data=None if self._val_data is None
            else {"in": self._val_data.get_inputs(),"out": self._val_data.get_outputs()},
            start_iteration=len(self._loss_history[str(val)]),
            inlet_profile=self._data.get_inlet_profile()
        )
        self._save_lbfgs_metrics(val)
        return pinn

    def _save_lbfgs_metrics(self, val: int | float) -> None:
        self._loss_history[str(val)].extend(self._lbfgs.get_loss_history(train_info=True))
        self._val_loss_history[str(val)].extend(self._lbfgs.get_loss_history(train_info=False))
        # First entry for key 'LBFGS' in dict
        if self._iter_time_history.get("LBFGS") is None:
            self._iter_time_history["LBFGS"] = {str(val): self._lbfgs.get_opt_time()}
        # Update underlying dict to key 'LBFGS'
        else:
            self._iter_time_history["LBFGS"].update({str(val): self._lbfgs.get_opt_time()})
        loss_comps: dict[str, list[float]] = self._lbfgs.get_loss_comps(train_info=True)
        for loss_name, loss_vals in loss_comps.items():
            self._loss_comps_history[loss_name] += loss_vals
        val_loss_comps: dict[str, list[float]] = self._lbfgs.get_loss_comps(train_info=False)
        for loss_name, loss_vals in val_loss_comps.items():
            self._val_loss_comps_history[loss_name] += loss_vals

    def equation_info(self) -> str:
        info: str = ""
        info += f"{dict_to_str(self._equations)}\n"
        return info

    def adam_info(self, val: int | float, include_val: bool) -> str:
        info: str = ""
        info += f"iters: {self._adam.get_iters(train_info=True)}\n"
        info += f"val_iters: {self._adam.get_iters(train_info=False)}\n"
        info += f"target_val: {self._adam.get_target_val()}\n"
        info += f"learning_rate: {self._adam.get_learning_rate():.1e}\n"
        info += f"batch_size: {self._adam.get_batch_size()}\n"
        info += f"val_interval: {self._adam.get_val_interval()}\n"
        info += f"train_time: {self._adam.get_opt_time()}\n"
        if include_val:
            info += f"varying_param {self._varying_param}: {val}\n" if self._varying_param != "" else ""
        else:
            info += f"varying_param: {self._varying_param}\n"
        info += f"data_points: {len(self._data.get_train_data()['in'])}\n"
        return info

    def lbfgs_info(self) -> str:
        info: str = ""
        info += f"iters: {self._lbfgs.get_iters(train_info=True)}\n"
        info += f"val_iters: {self._lbfgs.get_iters(train_info=False)}\n"
        info += f"target_val: {self._lbfgs.get_target_val()}\n"
        info += f"start_index: {self._adam.get_iters(train_info=True) + 1}\n"
        info += f"val_interval: {self._lbfgs.get_val_interval()}\n"
        info += f"train_time: {self._lbfgs.get_opt_time()}\n"
        info += f"data_points: {len(self._data.get_inputs())}\n"
        return info

    def _log_results(self, val: int | float, pinn: PINN) -> None:
        logger: Logger = Logger(self._model_path)
        logger.log(log_file="data_setup.txt", log_content=self._data.log_info())
        if self._val_data is not None:
            logger.log(log_file="val_data_setup.txt", log_content=self._val_data.log_info())
        logger.log(log_file="pinn_info.txt", log_content={
            "PINN": f"{pinn}",
            "Equations": self.equation_info(),
            "Terms and Scales:": dict_to_str(self._adam.get_terms_and_scales())}
                   )
        logger.log(log_file="loss_info.txt", log_content={
            "Adam": self.adam_info(val, True),
            "L-BFGS": self.lbfgs_info(),
            "Loss History": f"{self._loss_history[str(val)]}",
            "Validation Loss History": f"{self._val_loss_history[str(val)]}",
            "Loss Components History": dict_to_str(self._loss_comps_history),
            "Validation Loss Components History": dict_to_str(self._val_loss_comps_history)}
                   )
        if self._varying_param != "":
            logger: Logger = Logger(os.path.dirname(self._model_path))
            logger.log(log_file="data_setup.txt", log_content=self._data.log_info())
            if self._val_data is not None:
                logger.log(log_file="val_data_setup.txt", log_content=self._val_data.log_info())
            logger.log(log_file="pinn_info.txt", log_content={
                "PINN": f"{pinn}",
                "Equations": self.equation_info()}
                       )
            logger.log(log_file="loss_info.txt", log_content={
                "Adam": self.adam_info(val, False),
                "L-BFGS": self.lbfgs_info(),
                "Loss History": f"{dict_to_str(self._loss_history)}",
                "Validation Loss History": f"{dict_to_str(self._val_loss_history)}",
                "Adam Iteration Time History": dict_to_str(self._iter_time_history["Adam"]),
                "LBFGS Iteration Time History": dict_to_str(self._iter_time_history["LBFGS"]),
                "Validation Iteration Time History": dict_to_str(self._val_iter_time_history)}
                       )

    def plot_variation_losses(self) -> None:
        plot_variation_loss(
            log_path=os.path.dirname(self._model_path)
        )

    def plot_loss(self) -> None:
        plot_loss(
            loss_comps_history=self._loss_comps_history,
            val_loss_comps_history=self._val_loss_comps_history,
            adam_iters=self._adam.get_iters(train_info=True),
            lbfgs_iters=self._lbfgs.get_iters(train_info=True),
            val_adam_iters=self._adam.get_iters(train_info=False),
            val_adam_interval=self._adam.get_val_interval(),
            val_lbfgs_interval=self._lbfgs.get_val_interval(),
            adam_train_time=self._adam.get_opt_time(),
            lbfgs_train_time=self._lbfgs.get_opt_time(),
            save_path=self._model_path
        )