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

    args = parser.parse_args()

    if args.optimizer not in {'adam', 'sgd', 'nadam'}:
        parser.error("optimizer should be: adam, sgd, or nadam ")

    if args.activation not in {'relu', 'selu', 'sigmoid'}:
        parser.error("optimizer should be: relu, selu or sigmoid")

    if args.model not in {'cnn', 'alexnet', 'vgg', 'inceptionv3', 'resnet'}:
        parser.error("%s is not a known model" % args.model)
    try:
        args.repeats = int(args.repeats)
        assert(args.repeats > 0)
    except:
        parser.error("Repeats should be a positive integer")
    return args

def run_name(parameters, end):
    """ Create a name for the combination of parameters"""
    output_name = "{model}_{optimizer}_{activation}_{augment}_{end}".format(end=end, **parameters)
    return output_name
