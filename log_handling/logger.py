import os

import numpy as np

class Logger:
    def __init__(self, log_path: str, log_dir: str = "logs", ext: str = ""):
        self._log_path: str = os.path.join(log_path, log_dir) if log_dir != "" else log_path
        if ext != "":
            self._log_path = os.path.join(self._log_path, ext)

    def log(self, log_file: str, log_content: dict[str,str]) -> None:
        os.makedirs(self._log_path, exist_ok=True)
        with open(os.path.join(self._log_path, log_file), "w") as log_file:
            for category, content in log_content.items():
                log_file.write(log_object_styler(category, content))

    def save_arr(self, save_file: str, arr: np.ndarray) -> None:
        os.makedirs(self._log_path, exist_ok=True)
        np.save(os.path.join(self._log_path, save_file), arr)

def log_object_styler(category: str, content: str) -> str:
    if content.split("\n")[-1] != "":
        content += "\n"
    return f"**{category}\n{content}\n"

def dict_to_str(d: dict) -> str:
    ret_str: str = ""
    for key, val in d.items():
        ret_str += f"{key}: {val}\n"
    return ret_str

if __name__ == "__main__":
    pass