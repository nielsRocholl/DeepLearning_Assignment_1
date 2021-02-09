import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

# Models
from cnn_model import cnn_model

parser.add_argument("-d", "--dataset", type=str, default="cifar10", help="which dataset do you want to load, cifar10 or fashion_mnist?")

parser.add_argument("-o", "--optimizer", type=str, default="adam", help="which optimizer do you want to use?")

args = parser.parse_args()

if args.optimizer not in {'adam', 'sgd', 'sgdm', 'nadam'}:
    parser.error("optimizer should be: adam, sgd, sgdm or nadam ")


def load_data(data):
    # 50,000 32x32x3 color training images and 10,000 test images, labeled over 10 categories
    if data == 'cifar10':
        dataset = keras.datasets.cifar10
        shape = (32, 32, 3)
    # 60,000 images that are made up of 28x28x1 pixels, 10 categories
    # TODO: These images are too small for most architectures (pooling layers reduce it too much)
    if data == 'fashion_mnist':
        dataset = keras.datasets.fashion_mnist
        shape = (28, 28, 1)
    return dataset, shape


def main():
    args = parser.parse_args()
    # Load dataset
    dataset, shape = load_data(args.dataset)
    # Split into tetsing and training
    (train, train_labels), (val, val_labels) = dataset.load_data()
    # Divide rgb values by 255, which results in values between 0 and 1
    train = train / 255.0
    val = val / 255.0
    model = cnn_model(train, val, train_labels, val_labels, shape).AlexNet()


if __name__ == "__main__":
    main()

