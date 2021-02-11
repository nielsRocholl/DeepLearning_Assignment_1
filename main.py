import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from augment import augment_data

parser = argparse.ArgumentParser()

# Models
from cnn_model import cnn_model

parser.add_argument("-o", "--optimizer", type=str, default="adam", help="which optimizer do you want to use?")

parser.add_argument("-a", "--activation", type=str, default="adam",
                    help="which actification function do you want to use?")

parser.add_argument("-aug", "--augment", type=str, default="False",
                    help="Do you want to use data augmentation?")

args = parser.parse_args()

if args.optimizer not in {'adam', 'sgd', 'nadam'}:
    parser.error("optimizer should be: adam, sgd, or nadam ")

if args.activation not in {'relu', 'selu', 'hard_sigmoid'}:
    parser.error("optimizer should be: relu, selu or hard_sigmoid")


def format_example(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Make sure that image has a right size
    image = tf.image.resize(image, [150, 150])
    return image, label


def main():
    (dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
        name='rock_paper_scissors',
        data_dir='tmp',
        with_info=True,
        as_supervised=True,
        split=[tfds.Split.TRAIN, tfds.Split.TEST],
    )
    train_examples = dataset_info.splits['train'].num_examples
    test_examples = dataset_info.splits['test'].num_examples

    input_size_original = dataset_info.features['image'].shape[0]
    input_shape_original = dataset_info.features['image'].shape

    input_size_reduced = input_size_original // 2
    input_shape_reduced = (
        input_size_reduced,
        input_size_reduced,
        input_shape_original[2]
    )
    # INPUT_IMG_SIZE = input_size_reduced
    input_shape = input_shape_reduced

    dataset_train = dataset_train_raw.map(format_example)
    dataset_test = dataset_test_raw.map(format_example)

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

    steps_per_epoch = train_examples // batch_size
    validation_steps = test_examples // batch_size

    model = cnn_model(steps_per_epoch, validation_steps, dataset_train_shuffled, dataset_test_shuffled, input_shape,
                      activation=args.activation, optimizer=args.optimizer).cnn()


if __name__ == "__main__":
    main()
