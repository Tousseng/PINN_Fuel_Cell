from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:25:00 2024

@author: Madadi

Modiefied on Thu March 6 2025

@modifier: Sebastian
"""

import os

import pandas as pd
import tkinter as tk
import numpy as np

from tkinter import simpledialog, messagebox, filedialog
from typing import Callable, Literal


# Create a function to handle the GUI prompt for segment length
def get_segment_length() -> float | None | Callable:
    """
    Allows the user to input the segment length via a dialog-window.
    Returns:
        The length of a segment as a float.
    """

    length_input: str = simpledialog.askstring(
        "Input",
        "Do you want to divide each section in lengthwise segments? "
        "If so, enter the desired length (e.g., 0.05 meters) or type 'no' if no further splitting is desired."
    )
    
    if length_input is None:
        return None  # User canceled
    
    if length_input.lower() == 'no':
        return None  # No segmentation
    
    try:
        return float(length_input)  # Convert input to float
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number for the segment length.")
        return get_segment_length()  # Retry getting the input

# Function to segment the data along the dominant dimension
def divide_dataset(data: pd.DataFrame, segment_length: float) -> list[pd.DataFrame]:
    """
    Divides the dataset based on the largest value range in the x-, y- or z-coordinate.
    Args:
        data: DataFrame to be divided
        segment_length: length of each segment

    Returns:
        List of DataFrames that make up the initial DataFrame that was provided.
    """

    if data.empty:
        return []  # No points to segment

    coords = data.iloc[:, -3:]  # Get the x, y, z coordinates (last three columns, hence the -3:)
    lengths: pd.DataFrame = coords.max() - coords.min() # Range in each dimension
    
    if lengths.isnull().any():
        raise ValueError("Calculated lengths contain NaN values, check your input data.")
    
    major_dim_name = lengths.idxmax()  # This will return the column name
    min_val, max_val = coords[major_dim_name].min(), coords[major_dim_name].max()
    
    if (max_val - min_val) < segment_length:
        print(f"Segment length {segment_length} is larger than the available range in {major_dim_name}.")
        return [data]  # Return the entire dataset as a single segment

    boundaries = [min_val + i * segment_length for i in range(int((max_val - min_val) / segment_length) + 2)]
    
    segments = []
    for i in range(len(boundaries) - 1):
        lower_bound, upper_bound = boundaries[i], boundaries[i + 1]
        
        segment_data = data[(data[major_dim_name] >= lower_bound) & (data[major_dim_name] < upper_bound)]
        
        if not segment_data.empty:
            segments.append(segment_data)

    return segments

# Main processing function
def process_csv_files() -> None:
    """
    Main function handling the choice of the input data as well as calling the necessary functions.
    Returns:

    """

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    csv_files: Literal[""] | tuple[str, ...] = filedialog.askopenfilenames(title="Choose CSV files for segmentation")

    # Ask once for the segment length before processing any files
    segment_length: float = get_segment_length()

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        # Load the CSV data
        data: pd.DataFrame = pd.read_csv(csv_file, delimiter=',', skiprows=0)
        
        # Create a directory for the current CSV file
        output_folder: str = os.path.splitext(csv_file)[0]  # Folder named after the CSV file (without extension)
        os.makedirs(output_folder, exist_ok=True)
    
        # Create the new output filename based on the parent name and box index
        base_filename = csv_file.split(sep="/")[-1].split(".")[0] # Get the actual file name
        output_filename = os.path.join(output_folder, f'{base_filename}.csv')  # New naming convention

        # Save the points to a new CSV file in the corresponding folder
        data.to_csv(output_filename, sep=',', index=False)
        print(f"Extracted {len(data)} points to {output_filename}")

        data = data.select_dtypes(include=[np.number])  # Keep only numeric columns

        if segment_length is not None:
            segments: list[pd.DataFrame] = divide_dataset(data, segment_length)
            for i, segment in enumerate(segments):
                segment_filename: str = f'{base_filename}_segment_{i + 1}.csv'  # Only filename here
                saving_path = os.path.join(output_folder, segment_filename)
                segment.to_csv(saving_path, index=False)  # Save in the current output_folder
                print(f"Saved segment {i + 1} of length {segment_length} meters to {segment_filename}")
            with open(os.path.join(output_folder, "info.txt"), "w") as f:
                f.write(f"Segment length: {segment_length} m")
        else:
            print(f"No points found in {csv_file}")

    print("All files processed.")

if __name__ == '__main__':
    process_csv_files()