# Comparing the performance of CNNclassifiers on a rock, paper, scissors dataset

This repository contains the code for a project comparing the performance of various CNN architectures and setups on the [rock, paper, scissors](http://www.laurencemoroney.com/rock-paper-scissors-dataset/) dataset. The code consists of several Python scripts that can be run from the command line and some Python modules with functionality shared between the scripts. 

## Requirements
The code was tested with Tensorflow version 2.4.1, and requires the tensorflow_datasets package as well as numpy and matplotlib. 

## Overview
Models are defined in `cnn_model.py`. Logic related to storing and loading of data is in `utils.py` and `analysys.py`. The data augmentation function is in `augment.py`. 

## Usage
There are three runnable scripts: `train.py`, `make_plots.py` and `grid_plot.py`. To ensure proper imports without distributing the code as a python package, scripts should be run with the root of the repository as a working directory. All three scripts use the same set of parameters, which are used to construct a subdirectory for each setup where training data is stored when training and later retreived for further analysis by the other scripts. These parameters are defined in `utils.py` and can be shown by calling any of the scripts with `--help`:

```
usage: train.py [-h] [-o OPTIMIZER] [-a ACTIVATION] [-aug AUGMENT] [-m MODEL] [-out OUTPUT_PATH]
                    [-r REPEATS] [-e EPOCHS] [-l ARG_LISTS] [-p PLOT_PATH] [-v PLOT_VAR] [-xmax X_MAX]
                    [-ymax Y_MAX] [-pw PLOT_WIDTH] [-ph PLOT_HEIGHT]

optional arguments:
  -h, --help            show this help message and exit
  -o OPTIMIZER, --optimizer OPTIMIZER
                        which optimizer do you want to use?
  -a ACTIVATION, --activation ACTIVATION
                        which actification function do you want to use?
  -aug AUGMENT, --augment AUGMENT
                        Do you want to use data augmentation?
  -m MODEL, --model MODEL
                        Which model do you want to use?
  -out OUTPUT_PATH, --output_path OUTPUT_PATH
                        Where to save or read the training results?
  -r REPEATS, --repeats REPEATS
                        How many times to repeat the training? In combination with `--arg-lists True`
                        pass specific run nrs. e.g. -r 1,2,3.
  -e EPOCHS, --epochs EPOCHS
                        How many epochs to use for training. Default: use 100 epochs and early
                        stopping.
  -l ARG_LISTS, --arg_lists ARG_LISTS
                        If 'True': Accept comma-separated lists specifying multiple options for the
                        arguments model, optimizer, activation, augment, and repeats.
  -p PLOT_PATH, --plot_path PLOT_PATH
                        Where to save the plot. Only used when calling grid_plot.py
  -v PLOT_VAR, --plot_var PLOT_VAR
                        Variable to plot in the grid plot. Only used when calling grid_plot.py
  -xmax X_MAX, --x_max X_MAX
                        Number of epochs on the x-axis in the grid plot. Only used when calling
                        grid_plot.py
  -ymax Y_MAX, --y_max Y_MAX
                        Maximum of the y-axis. Only used when calling grid_plot.py
  -pw PLOT_WIDTH, --plot_width PLOT_WIDTH
                        Plot width in inches. Only used when calling grid_plot.py
  -ph PLOT_HEIGHT, --plot_height PLOT_HEIGHT
                        Plot height in inches. Only used when calling grid_plot.py
```

The `train.py` script trains the model on a set of setups and stores a snapshot of the best model, the training history, and the confusion matrix of the best model in a specified location. To reproduce the dataset that was used in the accompanying report, we would call the script as follows:


```bash
python3 assignment1/train.py -m "lenet,alexnet,vgg,resnet" -aug "False,True" -o "adam,sgd" -a "relu,sigmoid" -r "0,1,2" -l True -out /path/to/data
```

The grid plots from the reports can be generated as follows (for the `alexnet` architecture):

```bash
python3 assignment1/grid_plot.py -m "alexnet" -aug "False,True" -o "adam,sgd" -a "relu" -r "0,1,2" -l True -out /path/to/data -p "alexnet_loss.pdf" -pw 5 -ph 2.5  -v loss -ymax 6
%run assignment1/grid_plot.py -m "alexnet" -aug "False,True" -o "adam,sgd" -a "relu" -r "0,1,2" -l True -out /path/to/data -p "alexnet_acc.pdf" -pw 5 -ph 2.5  -v accuracy -ymax 1
```

We can also easily generate larger or smaller grids by varying the sets of comma-separated values for the parameters. E.g. we could use `-m "lenet,alexnet,vgg,resnet` to make one large grid plot for all models. 

This script will also generate the table with the best and worst accuracy for each setup (with a name constructed by appending `_table.txt` to the plot name). 

To make the confusion matrix plots we can call `make_plots.py` for a single setup, e.g. as follows when there are three runs for this setup:

```bash
python3 assignment1/make_plots.py -m "alexnet" -aug "False" -o "sgd" -a "relu" -r "3" -out /path/to/data 
```
The plots will be stored in the data directory. 
