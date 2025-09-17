# PINN - Fuel Cell
This repository is part of the data publication [Python Model PINN Fuel Cell v1.0.0](https://doi.org/10.5281/zenodo.17131101). It was developed by the following people:
- Julian Toussaint (**Corresponding Author**), 
[Send email](mailto:Toussaint@tme.rwth-aachen.de?subject=Code%20of%20PINN%20-%20Fuel%20Cell)
- Sebastian Pieper (**Code engineer**), 
[Send email](mailto:sebastian.pieper@rwth-aachen.de?subject=Code%20of%20PINN%20-%20Fuel%20Cell)

## Installing the Environment
The repository includes an `environment.yml` file that can be used to install your virtual environment with *Conda*. 
It includes all the necessary packages to run the code on the CPU or GPU.
<br>
To create the *Conda* environment, open the *conda prompt* and enter 
```
    conda env create -f environment.yml
```
The name of the environment will automatically be "PINN". If you desire a different name, add `--name <desired_name>` 
after the `create` keyword in the creation command above.
Further information about *Conda*, its installation, and the management of its environments can be found at the official 
[*Conda* website](https://docs.conda.io/projects/conda/en/latest/index.html).

## Working with the repository
This repository consists of two main parts. 
The first includes the code necessary for training the neural network models (sections "Settings for training the 
models" and "Steps to train a model").
The second part handles the prediction process of a trained model (sections "Settings for predicting with models" and 
"Steps to predict with a model").
If the first model is to be trained, preprocessing might be needed. The possibilities for that are explained in
"Preprocessing of data".

### Preprocessing of data
Before starting the training process, two different preprocessing steps may be wanted or needed.
<br>
**Data segmentation**
<br>
Firstly, the data can be segmented by using [`segment_data.py`](./preprocessing/segment_data.py). Using the segmented 
data instead of the full data set reduces the number of data points the neural network has to process. The segmentation 
is done in the direction with the largest value range.
<br>
After starting this script, a window opens that allows the user 
to select the desired data set. Afterward, a second window lets the user type in the desired length of the segment.
All segments are then created and saved in a folder inside the same directory as the initially chosen data set. The
name of the folder is identical to the data sets name.
<br>
**Data combination**
<br>
Secondly, different data sets can be combined into a single one using 
[`combine_data.py`](./preprocessing/combine_data.py). When using this script, an additional column containing the
channel widths will be added to the data set. If a single data set is used with this script, the same data set is 
created but with the additional column of its singular channel width.
<br>
Combining two data sets with different channel widths is necessary if a neural network capable of predicting data sets
with different channel widths is wanted. For such a network, single data sets with their channel widths' column are 
needed for postprocessing (use script on single data set).
<br>
Prior to starting the script, the `base_dir_name` in the first line below `if __name__ == "__main__":` has to be set.
With that, the parent directory of the "combined" directory, containing the combined data, is chosen.
This ensures that the files are later saved at <path_to_base_dir_name/base_dir_name/combined/>.
<br>
After starting the script, a window opens allowing the user to select multiple files (data sets). Note that the paths of the 
selected files need to include the `base_dir_name`, otherwise an error is thrown. The selected files are subsequently
combined and the column for the channel widths is added. They are named according to the selected files including the values of
their channel widths and saved.
<br>
Unsurprisingly, the name of the selected files needs to contain their channel width. A valid file name would be 
"Cathode_CL_values_1_0mm_segment_3.csv". The most important part is "1_0mm" which indicates a channel width of 1.0 mm.
This format for the channel width indication ("1_0mm" or "0_4mm") has to be used to ensure that the script works.
The prior part ("Cathode_CL_values") and subsequent part ("segment_3") can be adapted to display the underlying data 
correctly.

### Settings for training the models
To start the training, head to [`start/start.py`](./start/start.py). Most of the training process is automated, yet some settings 
are necessary to be applied by the user.
<br>
Firstly, the `network` needs to be specified. Here, the four options `PINN-CL`, `PINN-GC`, `ANN-CL`, and `ANN-GC` are 
available. "PINN" and "ANN" stand for "Physics-informed Neural Network" and "Artificial Neural Network", respectively. 
"CL" indicates that the model is trained based on data for the "Catalyst Layer" and, similarly, "GC" indicates the 
usage of "Gas Channel" data. Entering something else does not work, since the necessary internal settings are based on 
the supplied `network`.
<br>
The `msg_CL` and `msg_GC` variables allow the user to capture a message inside a `readme.txt` in the models' directory.
<br> 
To automatically train multiple models, possibly with variations in hyperparameters, the `varying_params` variable can be supplied with a combination of options that are listed above the variable inside 
[`start/start.py`](./start/start.py). 
The values of each option can be found in the first line of [`start/training_process.py`](./start/training_process.py). Their name begins with 
the name of the respective option and ends in "options", e.g. `epochs_options`. 
If a single model shall be trained, use `varying_params = [""]`.
<br>
Despite the settings for `varying_params`, each model needs further metrics. They can be found in 
[`start/networks_settings.py`](./start/network_settings.py). These include the metrics for the neural network (`pinn_metrics`), i.e. the number 
of layers `num_layers` and neurons per layer `num_neurons` as well as the metrics for the optimizers "Adam" and 
"L-BFGS". 
Their metrics are the number of iterations `iterations`, a threshold `target_val` that stops training if the total loss 
is less than `target_val` (single iteration of "L-BFGS" is done if stop happens in "Adam"), and the interval 
`val_interval` between two predictions of the neural network based on the validation data. "Adam" includes two 
additional metrics, namely the `learning_rate` that scales the loss gradient in backpropagation and the `batch_size` 
that determines the size of the mini-batches.
<br>
Inside [`start/start.py`](./start/start.py) are three more settings that need to be mentioned. Firstly, the keyword `predictor` 
inside the `Setup` constructor allows to specify predictors. They are added to the list as strings and are needed if
the neural network is supplied with data sets having different values of the predictor during training. 
At the moment, only the predictor of the channel width "b" is available. 
Furthermore, "b" is only needed to model the catalyst layer (`network = "PINN-CL"` or `network = "ANN-CL"`). 
To include others, adaptations to the internal translation [`utilities/internal_translation.py`](./utilities/internal_translation.py) and inside the 
training code [`training/training.py`](./training/training.py) would be necessary. 
Further information on the modeling approach is supplied in [`model_equations.md`](./model_equations.md).
<br>
Secondly, regarding the channel width, a `single_width` might be necessary to be set if a neural network of the catalyst
layer shall be trained and the data set used for training does not include a channel width already.
The value of `single_width` needs to be from the possible options of `channel_widhts_ref` in 
[`training/parameters.py`](./training/parameters.py).
<br>
Lastly, `save_iters` allows the user to specify a sequence of iterations at which interim models should be saved by 
supplying these iterations as a list.

### Steps to train a model
The steps to train a model are as follows:
1. Navigate to [`start/start.py`](./start/start.py).
2. Adapt the settings according to the previous section ("Settings for training the models").
3. Run [`start/start.py`](./start/start.py).
4. After a short period, a window will open asking you to choose the data set for training. This has to be a "csv" file.
5. If `target_val` is not set to `None` for any of the optimizers in [`start/networks_settings.py`](./start/network_settings.py), a second 
window opens to select the validation data ("csv" file again). 
Afterward, the training runs automatically. During training, information, such as the loss for each iteration and the 
time per iteration is displayed on the console. Furthermore, the values of some quantities are displayed too.

After the training is finished, the model is automatically saved in a directory inside "trained_model_directory". 
Additionally, plots of the loss over the number of iterations can be found in the models' directory as well.

## Settings for predicting with models
The entry point for predicting is [`postprocessing/predict.py`](./postprocessing/predict.py). The prediction revolves around comparing the 
results of the neural network against the CFD data. This is done using color plots, parity plots, histograms or averages 
of the data set. The settings are directly applied in the `Predict` constructor at the end of the file (below the 
`if __name__ == "__main__":` part).
<br>
Two different setups for the prediction already exist. They are toggled by switching `single` from `True` to `False` and 
vice versa. Here, `single = True` refers to models that were trained without multiple channel widths (gas channel models 
and catalyst layer models without multiple channel widths). Accordingly, the use case for `single = False` is when a 
catalyst layer model trained with multiple channel widths shall be used for the prediction process.
<br>
Firstly, the `model_path` does not need to be changed. It uses the path that is selected by the user via the opened 
window when starting `predict.py`. In the further explanations, this path will be referred to as *selected-directory*.
<br>
Firstly, setting `sane_file = True` (defaults to `True` if not supplied) will use the same CFD file that was used for 
training to compare to. Conversely, a model needs to be selected if `same_file = False`. This is the case when a 
model of the catalyst layer trained on different channel widths is chosen.
<br>
The different options to compare the predicted values of the neural network against the CFD data can be selected by 
`avgs` and `plots`. Setting `avgs = True` will compare based on averages of the data sets. For that, the variables for 
comparison need to be supplied via `var_list` when calling the `prediction` object. This option is, however, only 
intended when different channel widths for the catalyst layer were used during training. Otherwise, it should be `False`. 
The option `plots = True` creates the color plots, parity plots, and histograms for all outputs of the neural network 
and the CFD data set. The plots can be found in directories containing `figures` inside the *selected-directory*. 
Furthermore, the predictions of the neural network are saved in directories containing `csv_results` inside the 
*selected-directory*. Connected to the plotting is the option `save_data` which, when set to `True`, saves the results 
from transforming the data from an irregular mesh used for training to a regular mesh used for visualization. 
They are written to different files contained in `plot_info` inside the *selected-directory*. This option could be set 
to `False` if the transformed data already exists in `plot_info`.
<br>
To increase the users comfort, the `parent_directory` setting can be used to apply the predictions to multiple models 
which are grouped inside a directory. Setting `parent_directory = True` (defaults to `False` if not supplied) will 
therefore consider all models inside the *selected-directory* for the predictions.
Similarly, if `include_interim = True` (defaults to `False` if not supplied) is set, the predictions are applied to 
every interim model too.

### Steps to predict with a model
To predict with a saved model (by doing the steps in "Steps to train a model") is needed; then the following needs to be 
done:
1. Navigate to [`postprocessing/predict.py`](./postprocessing/predict.py).
2. Apply the desired settings in accordance with the previous section "Settings for predicting with models".
3. After hitting "Run" inside `Predict.py`, a window opens to select the directory (*selected-directory*) of the saved 
neural network model (or multiple saved network models, if `parent_directory = True` is set).
4. If `same_file` is set to `False`, a second window opens, that asks the user to choose the CFD data used for comparing 
the neural network against (necessary for a catalyst model trained on multiple channel widths).

The prediction happens automatically from there on. The results, such as plots or csv files can be found inside the 
*selected-directory* after the program is finished.

## Closing remarks
The plotting inside the repository is incorporated such that it can be done independently of the training and prediction 
process.
For that, the files inside `logs` or `plot_info` are necessary which are saved automatically inside the models directory during the 
training or prediction process, respectively.
For this independence, a logger [`log_handling/logger.py`](./log_handling/logger.py) and log parser [`log_handling/log_parser.py`](./log_handling/log_parser.py) were 
included.
Further information about the code structure of the repository and each module can be found in
[`repository_structure.md`](./repository_structure.md).
