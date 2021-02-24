import os
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

from augment import augment_data
from cnn_model import cnn_model
from utils import run_name, parse_arguments


def format_example(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Make sure that image has a right size
    image = tf.image.resize(image, [150, 150])
    return image, label

def load_dataset():
    """Load and pre-process the Rock, Paper, Scissors dataset"""
    (dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
        name='rock_paper_scissors',
        data_dir='tmp',
        with_info=True,
        as_supervised=True,
        split=[tfds.Split.TRAIN, tfds.Split.TEST],
    )

    # Convert input to float and resize to 150x150
    input_shape = (150, 150, 3)
    dataset_train = dataset_train_raw.map(format_example).cache()
    dataset_test = dataset_test_raw.map(format_example).cache()

    return ((dataset_train, dataset_test), input_shape)

def train_model(args, dataset, repeat, save=True):
    """Train a model based on the specified arguments. Optionally save the result. """
    ((dataset_train, dataset_test), input_shape) = dataset
    train_examples = len(dataset_train)
    test_examples = len(dataset_test)
    
    if args.augment == 'True':
        dataset_train = dataset_train.map(augment_data)

    batch_size = 32

    dataset_train_shuffled = dataset_train.shuffle(
        buffer_size=train_examples
    )

    dataset_train_shuffled = dataset_train_shuffled.batch(
        batch_size=batch_size
    )

    dataset_train_shuffled = dataset_train_shuffled.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    dataset_test_shuffled = dataset_test.batch(batch_size)

    steps_per_epoch = train_examples / batch_size
    validation_steps = test_examples / batch_size

    model = cnn_model(steps_per_epoch, validation_steps, dataset_train_shuffled, dataset_test_shuffled, input_shape, args.model, activation=args.activation, optimizer=args.optimizer, epochs = args.epochs)
    if args.model == 'cnn':
        model.cnn()
    if args.model == 'alexnet':
        model.AlexNet()
    if args.model == 'vgg':
        model.VGG()
    if args.model == 'lenet':
        model.LeNet()
    if args.model == 'resnet':
        model.ResNet()
    if save:
        # Construct a name for the output
        output_name = run_name(vars(args), repeat)
        output_path = os.path.join(args.output_path, output_name)
        print("Writing training results to: %s" % output_path)

        # Create output directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        model.save_final_model(output_path)    
        # Write model summary to output
        summary_file = os.path.join(output_path, 'summary.txt')
        with open(summary_file, "w") as summary:
          model.model.summary(print_fn = lambda x: summary.write(x + '\n'))
          #summary.write(model.model.summary())
if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Load the dataset
    dataset = load_dataset()

    # Train the model
    for rep in range(args.repeats):
        train_model(args, dataset, rep, True)
