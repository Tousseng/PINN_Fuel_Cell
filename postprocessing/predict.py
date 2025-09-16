from __future__ import annotations

from log_handling import Logger, DataLogParser, dict_to_str
from data_handling.data import Data
from log_handling.log_parser import PINNLogParser
from utilities.internal_translation import convert_var_name_rev, convert_var_name
from utilities.extraction import extract_info
from data_handling.coordinates import CoordinateFrame
from postprocessing.prediction_plots.data_preparation import calc_error_metrics, create_data_grids, denormalize_data
from postprocessing.prediction_plots import plot_cfd_pinn_comparison, plot_error_histogram, plot_error_scatter

import os
from typing import Literal

import tensorflow as tf
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

class Prediction:
    def __init__(self, model_path: str, parent_directory: bool = False, same_file: bool = True,
                 plots: bool = True, avgs: bool = True, save_data: bool = True,
                 include_interim: bool = False):
        self._model_path: str = model_path
        self._parent_directory: bool = parent_directory
        self._curr_model_path: str | None = None
        self._same_file: bool = same_file
        self._include_interim: bool = include_interim
        self._plots: bool = plots
        self._avgs: bool = avgs
        self._save_data: bool = save_data
        self._file_paths: Literal[""] | tuple[str, ...] = self._set_file_paths()

        self._fig_dir: str = "figures"
        self._csv_dir: str = "csv_results"
        self._ext: str = ""
        self._fig_path: str = ""
        self._csv_path: str = ""

        self._is_pinn: bool = True

        self._input_vars: list[str] = []
        self._inputs: np.ndarray = np.array([])
        self._output_vars: list[str] = []
        self._outputs: np.ndarray = np.array([])
        self._out_indices: dict[str,int] = {}
        self._norm_stats: dict[str,dict[str,float]] = {}
        self._flow_plane: list[str] = []

        self._df_cfd: pd.DataFrame = pd.DataFrame()
        self._df_pinn: pd.DataFrame = pd.DataFrame()

        self._lengths: dict[str,list[float]] = {}
        self._cfd_avgs: dict[str,list[float]] = {}
        self._nn_avgs: dict[str,list[float]] = {}

    def __call__(self, **kwargs) -> None:
        var_list: list[str] = self._output_vars
        try:
            var_list = kwargs["var_list"]
        except KeyError as key:
            print(f"Couldn't find key {key}")

        if self._parent_directory:
            for curr_model in os.listdir(self._model_path):
                # Only work with the data folders (not the 'logs') and not with files.
                if curr_model.endswith("pdf") or curr_model.endswith("svg") or curr_model == "logs":
                    continue
                self._curr_model_path = os.path.join(self._model_path, curr_model)
                print(f"Working on {self._curr_model_path}.")
                self._prediction_process(var_list)
                # Reset the lengths and averages
                self._lengths = {}
                self._cfd_avgs = {}
                self._nn_avgs = {}
        else:
            self._curr_model_path = self._model_path
            self._prediction_process(var_list)

    def _prediction_process(self, var_list: list[str]) -> None:
        for file_path in self._file_paths:
            self.update_fig_csv_dirs(file_path)
            self._retrieve_data(file_path)
            self._save_cfd_and_pinn_data(
                fields=self.get_fields(), model_path=self._curr_model_path,
                fig_path=self._fig_path, csv_path=self._csv_path
            )
            self._create_plots(
                model_path=self._curr_model_path, fig_dir_name=os.path.basename(self._fig_path)
            )
            self._surf_avg(var_list, file_path)
            if self._include_interim:
                interim_path: str = os.path.join(self._curr_model_path, "interim_models")
                for curr_interim in os.listdir(interim_path):
                    print(f"Working on {curr_interim}...")
                    curr_interim_path: str = os.path.join(interim_path, curr_interim)
                    interim_fig_path: str = os.path.join(curr_interim_path, "figures")
                    interim_csv_path: str = os.path.join(curr_interim_path, "csv_results")
                    self._retrieve_pinn_data(curr_interim_path)
                    self._save_cfd_and_pinn_data(
                        fields=self.get_fields(), model_path=curr_interim_path,
                        fig_path=interim_fig_path, csv_path=interim_csv_path
                    )
                    self._create_plots(
                        model_path=curr_interim_path, fig_dir_name=os.path.basename(interim_fig_path)
                    )
        self._save_surf_avgs(var_list)

    def _retrieve_data_info(self) -> DataLogParser:
        data_log_parser: DataLogParser = DataLogParser(self._curr_model_path)
        self._input_vars = data_log_parser.get_inputs()
        self._output_vars = data_log_parser.get_outputs()
        self._norm_stats = data_log_parser.get_norm_stats()
        return data_log_parser

    def _retrieve_cfd_data(self, file_path: str) -> None:
        data_log_parser: DataLogParser = self._retrieve_data_info()
        data: Data = Data(CoordinateFrame(flow_coord=data_log_parser.get_coords()["flow_coord"],
                                          flow_normal_coord=data_log_parser.get_coords()["flow_normal_coord"],
                                          flow_plane_normal_coord=data_log_parser.get_coords()["flow_plane_normal_coord"]
                                          ),
                          outputs=data_log_parser.get_outputs()
                          )
        data.load_data_from_log(data_log_parser, file_path)
        self._df_cfd = data.get_data_slice()
        self._inputs = data.get_inputs().astype("float32")
        self._outputs = data.get_outputs()
        self._out_indices = data.get_out_indices()
        self._flow_plane = data.get_coords().get_flow_plane()
        self._df_cfd = denormalize_data(self._df_cfd, self._norm_stats).reset_index(drop=True)

    def _retrieve_pinn_data(self, model_path: str) -> None:
        self._is_pinn = PINNLogParser(self._curr_model_path).is_pinn()
        # Loading the saved model from the full path
        loaded_pb_model = tf.saved_model.load(model_path)

        # Check the loaded model's signature
        # For instance, if the signature is 'serving_default'
        inference = loaded_pb_model.signatures['serving_default']

        # Get the predictions
        output_data = inference(tf.constant(self._inputs))

        # This is done because interference returns a dictionary
        output_data: tf.Tensor = output_data[list(output_data.keys())[0]]

        # converting the ndarray to a DataFrame with specified column names
        self._df_pinn = pd.DataFrame(output_data.numpy(), columns=self._output_vars)

        # This step is done to add the positions to the DataFrame of the PINN (predicted outputs).
        self._df_pinn[self._input_vars] = self._inputs

        # Denormalize the data
        self._df_pinn = denormalize_data(self._df_pinn, self._norm_stats)

    def _retrieve_data(self, file_path: str) -> None:
        self._retrieve_cfd_data(file_path)
        self._retrieve_pinn_data(self._curr_model_path)

    def _surf_avg(self, var_list: list[str], file_path: str) -> None:
        if not self._avgs:
            return None

        for var in var_list:
            field: str = convert_var_name_rev(var)
            length, part, segment = extract_info(file_path)
            cfd_vals: np.ndarray = self._df_cfd[field].to_numpy()
            pinn_vals: np.ndarray = self._df_pinn[field].to_numpy()
            if var not in list(self._lengths.keys()):
                self._cfd_avgs[var] = [cfd_vals.mean()]
                self._nn_avgs[var] = [pinn_vals.mean()]
                self._lengths[var] =  [length]
            else:
                self._cfd_avgs[var].append(cfd_vals.mean())
                self._nn_avgs[var].append(pinn_vals.mean())
                self._lengths[var].append(length)

    def update_fig_csv_dirs(self, file_path: str) -> None:
        if not self._same_file:
            self._ext = os.path.basename(file_path).split('.')[0]
        # Create directory to save plots
        self._fig_path = os.path.join(
            self._curr_model_path, f"{self._fig_dir}_{self._ext}" if self._ext != "" else self._fig_dir
        )
        self._csv_path = os.path.join(
            self._curr_model_path, f"{self._csv_dir}_{self._ext}" if self._ext != "" else self._csv_dir
        )

    def _set_file_paths(self) -> Literal[""] | tuple[str, ...]:
        file_paths: Literal[""] | tuple[str, ...] = ("",)
        if not self._same_file:
             file_paths = filedialog.askopenfilenames(title="Select the training data",
                                                            filetypes=[("CSV-Dateien", "*.csv")]
                                                            )
        # If no model path has been selected, display a message
        if not file_paths:
            raise Exception("No model selected. Please select a model.")
        return file_paths

    def get_file_paths(self) -> Literal[""] | tuple[str, ...]:
        return self._file_paths

    def get_fields(self) -> list[str]:
        return self._output_vars

    def _save_cfd_and_pinn_data(self, fields: list[str], model_path: str, fig_path: str, csv_path: str) -> None:
        if not self._save_data:
            return None

        self._create_dirs(fig_path, csv_path)

        logger: Logger = Logger(model_path, "plot_info", self._ext)

        grids_cfd: dict[str,np.ndarray] = create_data_grids(
            self._df_cfd, fields, self._flow_plane
        )
        grids_pinn: dict[str, np.ndarray] = create_data_grids(
            self._df_pinn, fields, self._flow_plane
        )

        for field in self._flow_plane + fields:
            logger.save_arr(f"{convert_var_name(field)}_CFD.npy", grids_cfd[convert_var_name(field)])
            logger.save_arr(f"{convert_var_name(field)}_{'PINN' if self._is_pinn else 'ANN'}.npy",
                            grids_pinn[convert_var_name(field)]
                            )

        logger.log(log_file="colorplot_info.txt", log_content={"Flow Plane": f"{self._flow_plane}",
                                                               "Fields": f"{fields}"})

        logger.log(log_file="hist_and_parity_info.txt",
                   log_content={convert_var_name(field):
                                    dict_to_str(calc_error_metrics(self._df_cfd, self._df_pinn, field)) for field in fields
                                }
                   )

        self._df_pinn.to_csv(
            os.path.join(csv_path, f"{'PINN' if self._is_pinn else 'ANN'}_predictions.csv"), index=False
        )
        self._df_cfd.to_csv(
            os.path.join(csv_path, f"CFD_values.csv"), index=False
        )

    def _var_info(self, var: str) -> str:
        info: str = ""
        info += f"b: {self._lengths[var]}\n"
        info += f"cfd: {self._cfd_avgs[var]}\n"
        info += f"{'PINN' if self._is_pinn else 'ANN'}: {self._nn_avgs[var]}\n"
        return info

    def _save_surf_avgs(self, var_list: list[str]) -> None:
        if not self._avgs:
            return None

        logger: Logger = Logger(self._curr_model_path, "")
        logger.log(log_file=os.path.join(self._curr_model_path, "surf_avgs.txt"),
                   log_content={f"{var}": self._var_info(var) for var in var_list}
                   )

    def _create_plots(self, model_path: str, fig_dir_name: str) -> None:
        if not self._plots:
            return None

        plot_cfd_pinn_comparison(model_dir=model_path,
                                 fig_dir_name=fig_dir_name,
                                 plot_extension=self._ext
                                 )
        plot_error_histogram(model_dir=model_path,
                             fields=self._output_vars,
                             fig_dir_name=fig_dir_name,
                             plot_extension=self._ext
                             )
        plot_error_scatter(model_dir=model_path,
                           fields=self._output_vars,
                           fig_dir_name=fig_dir_name,
                           plot_extension=self._ext
                           )

    @staticmethod
    def _create_dirs(fig_path: str, csv_path: str) -> None:
        try:
            os.mkdir(fig_path)
        except FileExistsError as e:
            print("Directory *figures* already exists. Existing figures will be overwritten.")
        try:
            os.mkdir(csv_path)
        except FileExistsError as e:
            print("Directory *csv_results* already exists. Existing csv-files will be overwritten.")

if __name__ == "__main__":
    # Open an explorer window to select the model
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file selection window and let the user select the model
    model_path = filedialog.askdirectory(title="Select the folder of the saved model")

    # If no model path has been selected, display a message
    if not model_path:
        raise Exception("No model selected. Please select a model.")

    # Toggle between the two 'Prediction' setups below
    single: bool = False

    if single:
        prediction: Prediction = Prediction(
            model_path=model_path,
            # Set to True if multiple models inside the same directory shall be predicted automatically
            parent_directory=False,
            # Set to True if predictions should be applied to interim models as well
            include_interim=False,
            same_file=True,
            avgs=False,
            plots=True,
            save_data=True
        )
    else:
        prediction: Prediction = Prediction(
            model_path=model_path,
            # Set to True if multiple models inside the same directory shall be predicted automatically
            parent_directory=False,
            # Set to True if predictions should be applied to interim models as well
            include_interim=False,
            same_file=False,
            avgs=True,
            plots=False,
            save_data=False
        )

    prediction(var_list=["eta", "j"])