import os
import numpy as np
import matplotlib.pyplot as plt
import itertools 

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
    confusion_list = np.round(confusion_list, 2)
    mean_confusion = np.round(mean_confusion, 2)
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

def plot_history(stats, save_path = None):
    """
    Plot a variable of the history
    """
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for idx, a in  enumerate(stats['history']):
        plt.plot(a['accuracy'], label='Training set ' + str(idx))
    for idx, a in  enumerate(stats['history']):
        plt.plot(a['val_accuracy'], label='Test set ' + str(idx), linestyle='--')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + 'accuracy.png')
    plt.clf()

    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for idx, a in  enumerate(stats['history']):
        plt.plot(a['loss'], label='Training set ' + str(idx))
    for idx, a in enumerate(stats['history']):
        plt.plot(a['val_loss'], label='Test set ' + str(idx), linestyle='--')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + 'loss.png')
    plt.clf()

    # plt.figure(figsize=(6, 4))
    # plt.show()
    # for h in stats['history']:
    #     plt.plot(h[variable])
    # plt.xlabel('epoch')
    # plt.ylabel(variable)
    # if save_path is None:
    #     plt.show()
    # else:
    #     plt.savefig(save_path)


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues, save_path = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size="x-large")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size="xx-large")
    plt.yticks(tick_marks, classes, size="xx-large" )

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black", size="x-large")

    plt.tight_layout()
    plt.ylabel('True label', size="xx-large")
    plt.xlabel('Predicted label', size="xx-large")
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.clf()  
