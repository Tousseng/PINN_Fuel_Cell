# Structure of the Repository
The repository is structured by packages. On the highest level, they are differentiated based on their general 
functionality, like data handling or training. If necessary, a package contains additional subpackages that bundle
functionality that belongs together. All modules inside the packages and subpackages of the repository are presented 
in the following:
- **data_handling** ==> Keeps track of the training and validation data. Furthermore, it contains functionality to
extract information about the data needed by other modules.
  - `coordinates.py`: *Used to distinguish between data sets with different alignments of their coordinate system.*
  - `data.py`: *Reads in the csv file of the data, normalizes it, and supplies needed information about the data to 
  other modules.*
- **log_handling** ==> Responsible for writing and reading log files.
  - `log_parser.py`: *Reads the logged files and transforms the information into python objects.*
  - `logger.py`: *Logs the results of the training for the subsequent prediction. Similarly, the results of the
  prediction gets logged for plotting.*
- **postprocessing** ==> Modules for plotting the results for the trained neural network.
  - loss_plots --> Plots concerning the loss curve (loss as a function of the iterations).
    - `fixed_parameter.py`: *Loss over iterations for each model. Shows the total loss and its components; created
    automatically after training.* 
    - `loss_over_time.py`: *Plot the final loss over the total training time. Is used for comparing the performance
    of hyperparameter settings.*
    - `settings.py`: *Additional settings specific to the loss plots.*
    - `varying_parameter.py`: *Loss over iterations for models with varying hyperparameters (multiple curves in the
    same plot); created automatically after training.*
  - predictions_plots --> Plots concerning the neural networks outputs (pressure, velocity, reaction rate, ...).
    - `cfd_params.py`: *Parameters of the catalyst layer CFD data set to transform the current density
    (reaction rate) from being related to the geometric area of the membrane to the area of the anode voltage
    collector.*
    - `color_plots.py`: *Plot the distribution of the neural networks predicted outputs compared to values of the
    CFD data on a 2D plane.*
    - `combined_plots.py`: *Combination of color plots for CFD data, PINN and ANN predictions, and a parity plot
    with both ANN and PINN predictions.*
    - `data_preparation.py`: *Transformation of irregular mesh data to regular grid data, denormalization, and
    calculation of error metrics for the histogram plots.*
    - `histogram_plots.py`: *Plot the histogram of the difference between the neural networks predicted outputs
    and the CFD data in a histogram. Mean and standard deviation are added as well.*
    - `parity_plots.py`: *Plot showing the neural networks predicted outputs as a function of the values of the
    CFD data.*
    - `settings.py`: *Additional settings specific to the prediction plots.*
    - `surf_avgs_plots.py`: *Plot the current density at the anode voltage collector as a function of the
    channel width. Plot for single and multiple current density curves are available.*
  - `colors.py`: *Includes all colors for the plots as a dict; their name are the keys and their
  RGB-values are the values of said dict.*
  - `general_settings.py`: *General settings for the plots such as text size, font, or size of the
  figures.*
  - `linestyles.py`: *Line styles to distinguish information in plots better.*
  - `predict.py`: ***Starting point** for the prediction program.*
- **preprocessing** ==> Investigate the training data set and preprocess the data set for training.
  - data_analysis --> Investigate the data sets before training to check for inconsistencies. For that, the physical 
  equations used for training are analyzed based on the supplied data set.
    - `data_points.py`: *Visualize the split of the data set into collocation and boundary points.*
    - `reaction_rate.py`: *Visualize the current density residual based on the modeling approach.*
    - `resistance.py`: *Visualize the electrical resistance model and get information about the
    models coefficients (can visualize the surface overpotential as well).*
    - `utils.py`: *Includes functionality to transform or visualize the results.*
  - `average_parameters.py`: *Average parameters for the physical equations over the entire supplied data
  set.*
  - `combine_data.py`: *Merge different data sets together; a column containing their channel widths will
  be added automatically. Can be used to add the channel width column to a single data set as well.*
  - `segment_data.py`: *Data sets (e.g. CFD data) can be split into segments of chosen size.*
- **start** ==> Modules to start training the neural networks and change the settings of the training.
  - `network_settings.py`: *Encapsulates all the metrics for the neural network and optimizers.*
  - `setup.py`: *Manages other modules needed for the training process and delegates the tasks.*
  - `start.py`: ***Starting point** of the training program.*
  - `training_process.py`: *Contains the steps in `setup.py` for the training. Furthermore, there are lists for the 
  variation of hyperparameters or training multiple models with the same hyperparameters.*
- **training** ==> Modules directly related to the training process.
  - `parameters.py`: *Separate file that contains all parameters used for training.*
  - `pinn.py`: *Neural network for training.*
  - `training.py`: *Train an ANN or PINN based on the specified equations. Gradients for the PDEs are determined here as 
  well.*
- **utilities** ==> Modules with general helper functionality that are used by other modules.
  - `cleanup.py`: *Delete the `__autograph_generated_file` files of the TensorFlow graphs created by python during 
  execution (currently not in use).*
  - `data_preprocessing.py`: *Only includes the function to split a data set into collocation and boundary points.*
  - `extraction.py`: *Extracts the length, part, and segment number information from the data set file name; not meant
  to be used directly. `predict.py` and `combine_data.py` use it, ensuring that all aforementioned information is
  included.*
  - `internal_trainslation.py`: *Translates the names of variables from the naming convention of *Star-CCM+*, internal 
  name for `training.py`, and the plots.*
  - `plotting.py`: *Includes functionality that is used by `fixed_parameter.py` and `varying_parameter.py` to 
  distinguish the training region of "Adam" and "L-BFGS" in the plots.*
  - `timing.py`: *Decorators for functions to time their duration.*
  - `unit_conversion.py`: *Convert units for the plotting.*
