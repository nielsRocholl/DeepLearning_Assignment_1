"""
Script to make plots and gather statistics from the training output
"""
import os
from utils import parse_arguments, run_name, iterate_args
import analysis
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from matplotlib.ticker import MaxNLocator

def loss_grid_plot(args, max_loss = 1.0, save_path=None):
    # Read plot parameters from arguments
    width = float(args.plot_width)
    height = float(args.plot_height)

    x_max = float(args.x_max)
    y_max = float(args.y_max)
    
    plot_var = args.plot_var

    # make plot grid
    augmented = len(args.augment)
    aug_names = {
        "True" : "Aug.",
        "False" : "Not aug."
    }
    models = len(args.model)
    model_names = {
        "alexnet" : "AlexNet",
        "lenet" : "LeNet-5",
        "cnn" : "CNN",
        "vgg" : "VGG",
        "resnet": "ResNet"
    }
    optimizers = len(args.optimizer)
    opt_names = {
        "adam" : "Adam",
        "sgd" : "SGD"
    }
    activations = len(args.activation)
    act_names = {
        "relu" : "ReLu",
        "sigmoid" : "Sigmoid"
    }

    var_names = {
        "loss" : "Loss",
        "acc" : "Accuracy"
    }
    
    rows = augmented * models
    cols = optimizers * activations
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(width,height), squeeze=False)
    for col in axes:
        for ax in col:
            ax.set_xticks(np.arange(0, x_max, 5))
            ax.set_xlim(0, x_max)
            ax.tick_params(axis='y', which='both', labelleft=ax.is_first_col(), labelright=ax.is_last_col(), left=True, right=True, direction="in")
            ax.tick_params(axis='x', direction="in")

            ax.set_ylim(0, y_max)

            ax.yaxis.set_major_locator(MaxNLocator(prune='upper', nbins=5))
            # Set y-axis label on the right
            if ax.is_last_col():
                ax.set_ylabel(var_names[plot_var],size=12)
                ax.yaxis.set_label_position("right")
            # Set x-axis label on the bottom
            if ax.is_last_row():
                ax.set_xlabel("Epoch",size=12)
    # Set the labels for model+augment
    for (n_m, model_key) in enumerate(args.model):
        for (n_a, aug_key) in enumerate(args.augment):
            ax = axes[n_m * augmented + n_a][0]
            ax.set_ylabel("{}\n{}".format(model_names[model_key], aug_names[aug_key]), size=13)
    # Set the labels for optimizer + activation
    for (n_a, act_key) in enumerate(args.activation):
        for (n_o, opt_key) in enumerate(args.optimizer):
            ax = axes[0][n_a * optimizers + n_o]
            ax.set_title("{}\n{}".format(act_names[act_key], opt_names[opt_key]), size=13)
    plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.05, hspace=0.05)
    # Plot the data into the subplots
    for (n_m, model_key) in enumerate(args.model):
        for (n_a, aug_key) in enumerate(args.augment):
            row = n_m * augmented + n_a
            for (n_a, act_key) in enumerate(args.activation):
                for (n_o, opt_key) in enumerate(args.optimizer):
                    col = n_a * optimizers + n_o
                    x = np.linspace(0,25)
                    y = np.linspace(0, 1)
                    run_args = copy(args)
                    run_args.model = model_key
                    run_args.augment = aug_key
                    run_args.activation = act_key
                    run_args.optimizer = opt_key
                    print(run_args)
                    for rep in run_args.repeats:
                        try:
                            history, confusion = analysis.load_result(run_args.output_path, vars(run_args), rep)
                            axes[row][col].plot(history[plot_var])
                            axes[row][col].plot(history['val_' + plot_var], ls=':')
                        except:
                            pass

    # Save the figure
    if not save_path is None:
        fig.savefig(save_path, bbox_inches="tight")


    
if __name__ == "__main__":
    args = parse_arguments()
    loss_grid_plot(args, save_path = args.plot_path)
    
