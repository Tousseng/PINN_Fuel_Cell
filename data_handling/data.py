from __future__ import annotations

"""
All the data for the training and the prediction is being handled in this module.
"""

from log_handling import DataLogParser, log_object_styler, dict_to_str
from .coordinates import CoordinateFrame
from utilities import convert_var_name, pred_trans, convert_var_list_rev

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, coord_frame: CoordinateFrame, outputs: list[str]):
        self._coords: CoordinateFrame = coord_frame
        self._info: dict[str,any] = {}
        self._data: pd.DataFrame = pd.DataFrame()
        self._slice_data: pd.DataFrame = pd.DataFrame()
        self._inlet_profile: pd.DataFrame = pd.DataFrame()
        self._norm_stats: dict[str,dict[str,float]] = {}
        self._inputs: list[str] = convert_var_list_rev(["x", "y", "z"])
        self._outputs: list[str] = outputs
        self._train_val_data: dict[str,dict[str,np.ndarray]] = {}

    def load_data(self, file_path: str, vel_lim: float, tol: float, slice_pos: float = None,
                  val_size: float = 0.2, rand: int = 12345, shuffle: bool = True) -> None:
        """
        Loads the data from the csv-file, given by *file_path*.
        Args:
            file_path: String to the file the data should be loaded from.
            vel_lim: Limit of the velocity that decides whether a data point can be considered for the inlet profile.
            tol: Thickness of the slice as a proportion of the distance between the boundary of the data and *slice_pos*.
            slice_pos: Position of the slice that is used to train the PINN.
            val_size: Proportion of validation data (e.g. 0.2 for 20%).
            rand: Random seed for reproducibility.
            shuffle:
        Returns:

        """
        if not self._data.empty:
            raise Exception("Object already contains data. No new data loaded.")

        self._info["file"] = file_path
        self._info["vel_lim"] = vel_lim
        self._info["tol"] = tol
        self._info["slice_pos"] = slice_pos
        self._info["val_size"] = val_size
        self._info["rand"] = rand

        self._data = pd.read_csv(self._info["file"], delimiter=',')
        # Shuffle the data
        if shuffle:
            self._data = self._data.sample(frac=1, random_state=rand).reset_index(drop=True)
        self._slice_data: pd.DataFrame = self._slice_dataframe(self._data, self._info["tol"], self._info["slice_pos"])
        self._set_inlet_profile(self._coords.get_flow_plane()[0])
        self._normalize_data()

    def load_data_from_log(self, data_log_parser: DataLogParser, new_file_path: str = "") -> None:
        data_metrics: dict[str,str] = data_log_parser.get_data_metrics()
        self._inputs = data_log_parser.get_inputs()
        self._norm_stats = data_log_parser.get_norm_stats()
        try:
            slice_pos: float | None = float(data_metrics["slice_pos"])
        except ValueError:
            slice_pos = None

        if new_file_path != "":
            file_path = new_file_path
        else:
            file_path = data_metrics["file"]

        self.load_data(file_path=file_path,
                       vel_lim=float(data_metrics["vel_lim"]),
                       tol=float(data_metrics["tol"]),
                       slice_pos=slice_pos,
                       val_size=float(data_metrics["val_size"]),
                       rand=int(data_metrics["rand"]),
                       shuffle=False
                       )

    def _normalize_data(self) -> None:
        if not self._norm_stats:
            for column in self._slice_data:
                if column == "inlet_velocity":
                    continue
                self._norm_stats[column] = {"mean": self._slice_data[column].mean(),
                                            "std":  self._slice_data[column].std()
                                            }
        for column in self._slice_data:
            if column == "inlet_velocity":
                continue
            mean: float = self._norm_stats[column]["mean"]
            std: float = self._norm_stats[column]["std"]
            # Dividing by zero not possible.
            if std > 0:
                # Normalization of the data
                self._slice_data.loc[:, column] = (self._slice_data[column] - mean) / std
                # Make sure that the current column can be found in inlet_profile as well.
                if column in self._inlet_profile.columns:
                    self._inlet_profile.loc[:, column] = (self._inlet_profile[column] - mean) / std

    def _denormalize_data(self) -> None:
        exclude_columns = ["inlet_velocity", "Density (kg/m^3)"]

        for column in self._slice_data:
            if column in self._norm_stats.keys() and column not in exclude_columns:
                mean = self._norm_stats[column]["mean"]
                std = self._norm_stats[column]["std"]
                self._slice_data = self._slice_data * std + mean

    def _slice_dataframe(self, data_frame: pd.DataFrame, tolerance: float, slice_position) -> pd.DataFrame:
        """
        Create a slice of the data in the plane_normal_direction.
        Args:
            data_frame: Input DataFrame to create a slice of.
            tolerance: Fraction of values between the minimum and maximum that should be used.
            slice_position:
        Returns:
            DataFrame slice of the input data with a thickness depending on tolerance located around the median.
        """

        # Create a thin slice in the direction normal to the desired plane (plane_normal_direction)
        plane_normal_values: pd.DataFrame = data_frame[self._coords.get_flow_plane_normal()]
        if slice_position is None:
            slice_position: float = plane_normal_values.median()

        if tolerance > 1 or tolerance < 0:
            raise ValueError("The tolerance may be chosen between 0.0 and 1.0!")

        # Define the range for filtering
        # The condition is used due to rounding errors
        # that leads to not all data points being included if tolerance = 1.0
        lower_bound: float = (
            slice_position - tolerance * (slice_position - plane_normal_values.min())
            if tolerance < 1.0
            else plane_normal_values.min()
        )

        upper_bound: float = (
            slice_position + tolerance * (plane_normal_values.max() - slice_position)
            if tolerance < 1.0
            else plane_normal_values.max()
        )

        # Filter the data
        sliced_data: pd.DataFrame = data_frame[(plane_normal_values >= lower_bound) & (plane_normal_values <= upper_bound)]
        return sliced_data.reset_index(drop=True)

    def _set_inlet_profile(self, coord_col: str) -> None:
        # Find the minimum value of the selected coordinate column
        min_value = self._slice_data[coord_col].min()

        # Define the tolerance
        tolerance_number = min_value * self._info["vel_lim"]
        tolerance_number = np.abs(tolerance_number)

        # Filter the rows where the coordinate value is within the range
        filtered_data = self._slice_data[(self._slice_data[coord_col] >= min_value)
                                   & (self._slice_data[coord_col] <= min_value + tolerance_number)]

        filtered_data = filtered_data.copy()

        self._inlet_profile = filtered_data[self._coords.get_flow_area() + self._outputs]

    def add_input_vars(self, var_list: list[str]) -> None:
        for var in var_list:
            self._inputs.append(var)

    def remove_input_vars(self, var_list: list[str]) -> None:
        for var in var_list:
            self._inputs.remove(var)

    def add_output_vars(self, var_list: list[str]) -> None:
        for var in var_list:
            self._outputs.append(var)

    def remove_output_vars(self, var_list: list[str]) -> None:
        for var in var_list:
            self._outputs.remove(var)

    def _split_data(self, val_fraction: float | None) -> None:
        """
        Divides the data into training and validation data.
        The output of 'train_test_split' is (x_train, x_test, y_train, y_test) with x representing the input variables
        and y representing the output variables.
        """
        test_size: float = self._info["val_size"] if val_fraction is None else val_fraction
        if test_size < 0:
            raise ValueError(
                f"Fraction of data set for validation may not be negative: test_size = {test_size}"
            )
        if test_size > 1:
            raise ValueError(
                f"Fraction of data set for validation may not be larger than one: test_size = {test_size}"
            )
        if test_size == 0:
            train_in = self.get_inputs()
            val_in = np.array([])
            train_out = self.get_outputs()
            val_out = np.array([])
        elif test_size == 1:
            train_in = np.array([])
            val_in = self.get_inputs()
            train_out = np.array([])
            val_out = self.get_outputs()
        else:
            train_in, val_in, train_out, val_out = train_test_split(
                self.get_inputs(), self.get_outputs(),
                test_size=test_size
            )
        self._train_val_data["train"] = {"in": train_in, "out": train_out}
        self._train_val_data["val"] = {"in": val_in, "out": val_out}

    def check_data_quality(self) -> None:
        pass

    def get_flow_dir(self) -> str:
        return self._coords.get_flow_plane()[0]

    def get_num_outputs(self) -> int:
        return len(self._outputs)

    def get_num_inputs(self) -> int:
        return len(self._inputs)

    ## NEEDS GENERALIZATION
    def get_inputs(self) -> np.ndarray:
        return self._slice_data[self._inputs].to_numpy()

    def get_outputs(self) -> np.ndarray:
        return self._slice_data[self._outputs].to_numpy()

    def get_input_vars(self) -> list[str]:
        return self._inputs

    def get_output_vars(self) -> list[str]:
        return self._outputs

    def get_inlet_profile(self) -> np.ndarray:
        return self._inlet_profile.to_numpy()

    def get_train_data(self, val_fraction: float | None = None) -> dict[str,np.ndarray]:
        if self._data.empty:
            raise Exception("Tried to retrieve data that wasn't loaded yet.")
        if not self._train_val_data:
            self._split_data(val_fraction)
        return self._train_val_data["train"]

    def get_val_data(self, val_fraction: float | None = None) -> dict[str,np.ndarray]:
        if self._data.empty:
            raise Exception("Tried to retrieve data that wasn't loaded yet.")
        if not self._train_val_data:
            self._split_data(val_fraction)
        return self._train_val_data["val"]

    def get_data_file(self) -> str:
        return self._info["file"]

    def get_coords(self) -> CoordinateFrame:
        return self._coords

    def get_norm_stats(self) -> dict[str,dict[str,float]]:
        return self._norm_stats

    def get_data_slice(self) -> pd.DataFrame:
        return self._slice_data

    def get_in_indices(self) -> dict[str,int]:
        in_indices: dict[str,int] = {}
        for idx, var in enumerate(self._inputs):
            min_val, max_val = self._get_data_range(var)
            if min_val == max_val or var in list(pred_trans.keys()):
                in_indices[convert_var_name(var)] = -1
            else:
                in_indices[convert_var_name(var)] = idx
        return in_indices

    def get_out_indices(self) -> dict[str,int]:
        out_indices: dict[str,int] = {}
        for idx, var in enumerate(self._outputs):
            out_indices[convert_var_name(var)] = idx
        return out_indices

    def _get_data_range(self, field: str) -> tuple[float,float]:
        min_val: float = np.min(self._slice_data[field].to_numpy())
        max_val: float = np.max(self._slice_data[field].to_numpy())
        return min_val, max_val

    def log_info(self) -> dict[str,str]:
        return {"CoordinateFrame": f"{self._coords}",
                "Data Metrics": dict_to_str(self._info),
                "Inputs": f"{self._inputs}",
                "Outputs": f"{self._outputs}",
                "Normalization": f"{dict_to_str(self._norm_stats)}"
                }

    def __str__(self) -> str:
        ret_str: str = ""
        ret_str += log_object_styler("CoordinateFrame", f"{self._coords}")
        ret_str += log_object_styler("Data Metrics", dict_to_str(self._info))
        ret_str += log_object_styler("Inputs", f"{self._inputs}")
        ret_str += log_object_styler("Outputs", f"{self._outputs}")
        ret_str += log_object_styler("Normalization", f"{dict_to_str(self._norm_stats)}")
        return ret_str

def test_module():
    pass

if __name__ == "__main__":
    test_module()