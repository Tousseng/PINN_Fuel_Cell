from __future__ import annotations

import os
from abc import ABC


class LogParser(ABC):
    def __init__(self, model_path: str):
        self._log_path: str = model_path
        self._info_dict: dict[str,list[str]] = {}
        self._log_ext: str = ""

    def _log_to_dict(self, string_list: list[str]) -> None:
        sep: str = "**"
        info_list: list[str] = []
        while len(string_list) != 0:
            s: str = string_list.pop().replace("\n", "")
            if sep in s:
                self._info_dict[s.replace("*", "")] = info_list
                info_list = []
                continue
            if s != "":
                if "['" in s:
                    info_list = s.split("'")[1::2]
                else:
                    info_list.append(s.replace("\n", ""))

    def read_log(self) -> None:
        info_list: list[str] = []
        with open(os.path.join(self._log_path, self._log_ext), "r") as f:
            for line in f.readlines():
                info_list.append(line)
        self._log_to_dict(info_list)

    def _get_info_dict(self, field: str) -> dict[str,str]:
        if not self._info_dict:
            self.read_log()
        fields: list[str] = self._info_dict.get(field)
        data_dict: dict[str, str] = {}
        for elem in fields:
            key, val = elem.split(":", 1)
            val = val.replace(" ", "", 1)
            data_dict[key] = val

        return data_dict

    def _get_info_list(self, field: str) -> list[str]:
        if not self._info_dict:
            self.read_log()
        fields: list[str] = self._info_dict.get(field)
        return fields

    @staticmethod
    def _str_dict_to_dict(str_dict: str) -> dict[str,float]:
        str_dict = (str_dict.replace("{", "").replace("}", "").
                    replace("'", "").replace(" ", "")
                    )
        mean, std = str_dict.split(",", 1)
        mean_name, mean_val = mean.split(":", 1)
        std_name, std_val = std.split(":", 1)
        return {mean_name: float(mean_val), std_name: float(std_val)}

    @staticmethod
    def _str_list_to_list(str_list: str, var_type: str = "", no_list: bool = False
                          ) -> list[str] | str | list[float] | float | list[int] | int:
        l: list[str] = (str_list.replace(" ", "").replace("[", "").replace("]", "").
                    split(",")
                    )
        l_ret: list[float] = []
        for val in l:
            if val == "":
                continue
            if var_type == "float":
                val = float(val)
            elif var_type == "int":
                val = int(val)
            if no_list:
                return val
            l_ret.append(val)
        return l_ret

    def get_info_dict(self) -> dict[str,list[str]]:
        return self._info_dict

    def get_log_path(self) -> str:
        return self._log_path

    def _transform_info(self, info: dict[str,str], is_list: bool, no_list: bool = False
                        ) -> dict[str,float | list[float] | dict[str,float]]:
        d_ret: dict[str,float | list[float] |dict[str, float]] = {}
        for var, vals in reversed(info.items()):
            d_ret[var] = self._str_list_to_list(vals, "float", no_list) if is_list \
                else self._str_dict_to_dict(vals)
        return d_ret

class DataLogParser(LogParser):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._log_path = os.path.join(self._log_path, "logs")
        self._log_ext = "data_setup.txt"

    def get_inputs(self) -> list[str]:
        return self._get_info_list("Inputs")

    def get_outputs(self) -> list[str]:
        return self._get_info_list("Outputs")

    def get_data_metrics(self) -> dict[str,str]:
        return self._get_info_dict("Data Metrics")

    def get_coords(self) -> dict[str,str]:
        return self._get_info_dict("CoordinateFrame")

    def get_norm_stats(self) -> dict[str,dict[str,float]]:
        return self._transform_info(self._get_info_dict("Normalization"), is_list=False)

class PINNLogParser(LogParser):
    def __init__(self, model_path: str, ext: str = ""):
        super().__init__(model_path)
        self._log_path = os.path.join(self._log_path, "logs")
        self._log_ext = "pinn_info.txt"

    def is_pinn(self) -> bool:
        d: dict[str, str] = self._get_info_dict("Equations")
        for val in list(d.values()):
            if val == "True":
                return True
        return False

class LossLogParser(LogParser):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._log_path: str = os.path.join(self._log_path, "logs")
        self._log_ext: str = "loss_info.txt"
        self._msg: str = "Validation "

    def get_loss_history(self, train_info: bool) -> dict[str,list[float]]:
        msg: str = "" if train_info else self._msg
        return self._transform_info(self._get_info_dict(f"{msg}Loss History"), is_list=True)

    def get_singular_loss_history(self, train_info: bool) -> list[float]:
        msg: str = "" if train_info else self._msg
        return self._str_list_to_list(self._get_info_list(f"{msg}Loss History")[0], var_type="float")

    def get_loss_comps_history(self, train_info: bool) -> dict[str,list[float]]:
        msg: str = "" if train_info else self._msg
        return self._transform_info(self._get_info_dict(f"{msg}Loss Components History"), is_list=True)

    def get_iter_time_history(self) -> dict[str,dict[str,float]]:
        adam_iter_history: dict[str,list[float] | float] =  self._transform_info(
            self._get_info_dict("Adam Iteration Time History"), is_list=True, no_list=True
        )
        lbfgs_iter_history: dict[str, float] = self._transform_info(
            self._get_info_dict("LBFGS Iteration Time History"), is_list=True, no_list=True
        )
        return {"Adam": adam_iter_history, "LBFGS": lbfgs_iter_history}

    def get_adam_iters(self) -> int:
        return int(self._get_info_dict("Adam")["iters"])

    def get_adam_val_iters(self) -> int:
        return int(self._get_info_dict("Adam")["val_iters"])

    def get_adam_val_interval(self) -> int | None:
        val_interval: str = self._get_info_dict("Adam")["val_interval"]
        return None if val_interval == "None" else int(val_interval)

    def get_adam_train_time(self) -> float:
        return float(self._get_info_dict("Adam")["train_time"])

    def get_varying_param(self) -> str:
        return self._get_info_dict("Adam")["varying_param"]

    def get_lbfgs_iters(self) -> int:
        return int(self._get_info_dict("L-BFGS")["iters"])

    def get_lbfgs_val_iters(self) -> int:
        return int(self._get_info_dict("L-BFGS")["val_iters"])

    def get_lbfgs_val_interval(self) -> int | None:
        val_interval: str = self._get_info_dict("L-BFGS")["val_interval"]
        return None if val_interval == "None" else int(val_interval)

    def get_lbfgs_train_time(self) -> float:
        return float(self._get_info_dict("L-BFGS")["train_time"])

class ColorLogParser(LogParser):
    def __init__(self, model_path: str, ext: str = ""):
        super().__init__(model_path)
        self._log_path = os.path.join(self._log_path, "plot_info")
        if ext != "":
            self._log_path = os.path.join(self._log_path, ext)
        self._log_ext = "colorplot_info.txt"

    def get_flow_plane(self) -> list[str]:
        return self._get_info_list("Flow Plane")

    def get_fields(self) -> list[str]:
        return self._get_info_list("Fields")

class HistLogParser(LogParser):
    def __init__(self, model_path: str, ext: str = ""):
        super().__init__(model_path)
        self._log_path = os.path.join(self._log_path, "plot_info")
        if ext != "":
            self._log_path = os.path.join(self._log_path, ext)
        self._log_ext = "hist_and_parity_info.txt"

    def get_cfd_arr(self, field: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)["data"],
                                      "float"
                                      )

    def get_pinn_arr(self, field: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)["data_predicted"],
                                      "float"
                                      )

    def get_diff(self, field: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)["diff"],
                                      "float"
                                      )

    def get_mean_deviation(self, field: str) -> float:
        return float(self._get_info_dict(field)["mean_deviation"])

    def get_mean_percent_deviation(self, field: str) -> float:
        return float(self._get_info_dict(field)["mean_percent_deviation"])

    def get_mse(self, field: str) -> float:
        return float(self._get_info_dict(field)["mse"])

    def get_std_deviation(self, field: str) -> float:
        return float(self._get_info_dict(field)["std_deviation"])

class AvgLogParser(LogParser):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._log_ext = "surf_avgs.txt"

    def get_channel_widths(self, field: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)["b"], "float")

    def get_cfd_avgs(self, field: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)["cfd"], "float")

    def get_nn_avgs(self, field: str, nn_name: str) -> list[float]:
        return self._str_list_to_list(self._get_info_dict(field)[f"{nn_name}"], "float")

    def get_network(self) -> bool:
        return bool(self._get_info_dict("Network")["PINN"])

def test_module():
    from tkinter import filedialog
    log_path: str = filedialog.askdirectory(title="Choose the folder of the log files.")
    pinn_log_parser: PINNLogParser = PINNLogParser(log_path)
    print(pinn_log_parser.is_pinn())

if __name__ == "__main__":
    test_module()