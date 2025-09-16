from utilities.extraction import extract_info

import os
from tkinter import filedialog

import pandas as pd

def combine_csv_files(dir_name: str) -> None:
    combined_data: pd.DataFrame = pd.DataFrame()
    channel_width: str = "Channel Width (m)"
    lengths: list[str] = []
    parts: list[str] = []
    segments: list[str] = []
    file_paths = filedialog.askopenfilenames(title="Choose CSV files to combine")
    if not file_paths:
        raise FileNotFoundError("No files were selected.")

    # Find the path that ends in <dir_name>
    base_path: str = os.path.dirname(file_paths[0])
    while os.path.basename(base_path) != dir_name:
        base_path = os.path.dirname(base_path)
        # Check if only root directory remains
        if base_path == os.path.dirname(base_path):
            raise FileNotFoundError(f"'{dir_name}' does not exist in '{os.path.dirname(file_paths[0])}'.")

    for file_path in file_paths:
        length, part, segment = extract_info(file_path)
        if len(segments) > 0 and len(parts) > 0:
            if segment != segments[-1] or part != parts[-1]:
                raise ValueError("File are not from the same segment or share the same part.")
        lengths.append(f"{str(length * 10 ** 3).replace('.','_')}mm")
        parts.append(part)
        segments.append(segment)
        data: pd.DataFrame = pd.read_csv(file_path)
        # Add column with channel length in each row
        data[channel_width] = [length for _ in range(data.shape[0])]
        if combined_data.empty:
            combined_data = data
        else:
            combined_data = pd.concat([combined_data, data], axis=0)

    save_path: str = os.path.join(base_path, "combined")
    os.makedirs(save_path, exist_ok=True)
    combined_data.to_csv(
        os.path.join(save_path, f"combined_{parts[-1]}{'-'.join(lengths)}{segments[-1]}.csv"),
        index=False
    )

if __name__ == "__main__":
    # Set the base directory where the combined data sets are supposed to be saved
    base_dir_name: str = "Cathode_Gas_Channel_W_Variation"
    combine_csv_files(dir_name=base_dir_name)