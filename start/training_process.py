from __future__ import annotations

from setup import Setup

############################################ USER SETTINGS #############################################################
# Define ranges for the varying parameters
model_options: list[int]= [num for num in range(1,21)]
epochs_options: list[int] = [200, 350, 500, 650]
iteration_options: list[int] = [25, 50, 100, 200]
batch_size_options: list[int] = [500, 1000, 1500, 2000]
learning_rate_options: list[float] = [1e-2, 1e-1, 1e0, 1e1]
layers_options: list[int] = [1, 2, 3, 4]
neurons_options: list[int] = [1, 3, 5, 7]
############################################ USER SETTINGS #############################################################

def perform_initial_training(file_path: str, setup: Setup, varying_params: list[str]):
    if not file_path:
        raise Exception("At least one file needs to be chosen.")

    setup.load_train_data(file_path=file_path)
    setup.load_val_data()
    for varying_param in varying_params:
        iter_var: list[int] | list[float] = []
        if varying_param == "model":
            iter_var = model_options
        elif varying_param == "epochs":
            iter_var = epochs_options
        elif varying_param == "iterations":
            iter_var = iteration_options
        elif varying_param == "batch_size":
            iter_var = batch_size_options
        elif varying_param == "learning_rate":
            iter_var = learning_rate_options
        elif varying_param == "layers":
            iter_var = layers_options
        elif varying_param == "neurons":
            iter_var = neurons_options
        else:
            iter_var.append(-42)

        for val in iter_var:
            setup.set_var(varying_param)
            setup.setup_pinn(val)
            setup.train_pinn(val)
            setup.plot_loss()
            setup.reset_training()
        if varying_param != "":
            setup.plot_variation_losses()
        setup.reset_history()