import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimizer", type=str, default="adam", help="which optimizer do you want to use?")

    parser.add_argument("-a", "--activation", type=str, default="relu",
                        help="which actification function do you want to use?")

    parser.add_argument("-aug", "--augment", type=str, default="False",
                        help="Do you want to use data augmentation?")

    parser.add_argument("-m", "--model", type=str, default="cnn",
                        help="Which model do you want to use?")

    parser.add_argument("-out", "--output_path", type=str, default="training_output/",
                        help="Where to save the training results?")

    parser.add_argument("-r", "--repeats", type=str, default="1",
                        help="How many times to repeat the training?")
    parser.add_argument("-e", "--epochs", type=str, default="es",
                        help="How many epochs to use for training. Default: use 100 epochs and early stopping.")
    args = parser.parse_args()

    if args.optimizer not in {'adam', 'sgd', 'nadam'}:
        parser.error("optimizer should be: adam, sgd, or nadam ")

    if args.activation not in {'relu', 'selu', 'sigmoid'}:
        parser.error("activation should be: relu, selu or sigmoid")

    if args.model not in {'cnn', 'alexnet', 'vgg', 'lenet', 'resnet'}:
        parser.error("%s is not a known model" % args.model)
    if args.epochs != 'es':
        try:
            args.epochs = int(args.epochs)
            assert(args.epochs > 0)
        except:
            parser.error("epochs is %s, should be 'es' or a positive integer." % args.epochs)
            
    try:
        args.repeats = int(args.repeats)
        assert(args.repeats > 0)
    except:
        parser.error("Repeats should be a positive integer")
    return args

def run_name(parameters, end):
    """ Create a name for the combination of parameters"""
    if parameters['epochs'] != 'es':
        epoch_str = "_%depochs" % parameters['epochs']
    else:
        epoch_str = ""
    output_name = "{model}_{optimizer}_{activation}_{augment}{epoch}_{end}".format(end=end, epoch=epoch_str, **parameters)
    return output_name
