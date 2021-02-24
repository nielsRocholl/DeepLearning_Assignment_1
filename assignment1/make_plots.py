"""
Script to make plots and gather statistics from the training output
"""
import os
from utils import parse_arguments, run_name
import analysis

if __name__ == "__main__":
    args = parse_arguments()
    stats = analysis.load_repeats(args.output_path, vars(args), args.repeats)

    cm_plot_labels = ['rock','paper', 'scissors']
    cm = stats['mean_confusion']
    save_path = os.path.join(args.output_path, run_name(vars(args), 'confusion' + ".png"))
    analysis.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix', save_path=save_path)
    # for var in ["accuracy", "loss"]:
    #     save_path = os.path.join(
    #         args.output_path,
    #         run_name(vars(args), var + ".png")
    #     )
    save_path = os.path.join(args.output_path, run_name(vars(args), ".png"))
    analysis.plot_history(stats, save_path = save_path)
