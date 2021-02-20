import os
import numpy as np
import matplotlib.pyplot as plt

from utils import run_name

def load_result(path, parameters, repeat):
    run_path = os.path.join(path, run_name(parameters, repeat))
    history_path = os.path.join(run_path, 'history.npy')
    confusion_path = os.path.join(run_path, 'confusion.npy')
    
    history = np.load(history_path, allow_pickle=True)[()]
    confusion = np.load(confusion_path, allow_pickle=True)[()]
    return history, confusion

def load_repeats(path, parameters, num_repeats):
    """ Aggregate results from several runs. """
    history_list = []
    confusion_list = []
    best_epochs = []
    best_val = []
    best_train = []
    for i in range(num_repeats):
        history, confusion = load_result(path, parameters, i)
        history_list.append(history)
        confusion_list.append(confusion)

        # Determine the epoch where validation accuracy was best
        best_epoch = np.argmax(history['val_accuracy'])
        best_epochs.append(best_epoch)
        best_val.append(history['val_accuracy'][best_epoch])
        # Training accuracy at the epoch with best validation accuracy
        best_train.append(history['accuracy'][best_epoch])
    mean_confusion = np.mean(confusion_list, axis=0)
    stats = {
        'history' : history_list,
        'confusion' : confusion_list,
        'mean_confusion' : mean_confusion,
        'best_epochs' : best_epochs,
        'best_val_acc' : best_val,
        'best_train_acc' : best_train,
        'parameters' : parameters
    }
    return stats

def plot_history(stats, variable = 'val_accuracy', save_path = None):
    """
    Plot a variable of the history
    """
    plt.figure(figsize=(6, 4))
    for h in stats['history']:
        plt.plot(h[variable])
    plt.xlabel('epoch')
    plt.ylabel(variable)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    
