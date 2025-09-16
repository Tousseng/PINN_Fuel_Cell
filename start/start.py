from __future__ import annotations

from setup import Setup
from data_handling.data import Data
from utilities import convert_var_list_rev
from training_process import perform_initial_training
from network_settings import set_coordinates, set_outputs, set_metrics, set_equations

from tkinter import filedialog

msg_CL: str = """\
"""

msg_GC: str = """\
"""

# Choose the setup ("PINN-CL", "PINN-GC", "ANN-CL", or "ANN-GC")
network: str = "PINN-CL"

# Possible varying parameters:
#   ""                  -> No variation study
#   "model"             -> Variation study of the model itself; no parameters are changed
#   "epochs"            -> Variation study of epochs (Adam)
#   "iterations"        -> Variation study of iterations (L-BFGS)
#   "batch_size"        -> Variation study of batch size (Adam)
#   "learning_rate"     -> Variation study of learning rate (Adam)
#   "layers"            -> Variation study of hidden layers
#    "neurons"          -> Variation study of neurons per hidden layer

varying_params: list[str] = [""]

setup: Setup = Setup(
    data = Data(
        coord_frame=set_coordinates(network),
        outputs=set_outputs(network)
    ),
    metrics=set_metrics(network),
    model_dir="../trained_model_directory",
    # Use "b" inside the square brackets below to include the channel width as a predictor
    predictors=convert_var_list_rev([]),
    equations=set_equations(network),
    msg=msg_CL if network == "PINN-CL" else msg_GC if network == "PINN-GC" else "",
    # Supply a channel width below if necessary, e.g. 6.87388e-4
    single_width=None,
    # Save the neural network at certain iterations by supplying save_iters with a list of iterations to save the model
    # at, e.g. save_iters=[10, 20, 50, 100, ...] to save it at iteration 10, 20, 50, 100, ...
    save_iters=None,
    # If detailed information about the training is desired, set debug=True
    debug=False
)

# File selection
file_path: str = filedialog.askopenfilename(title="Choose a CSV file for training")

perform_initial_training(file_path=file_path, setup=setup, varying_params=varying_params)