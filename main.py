import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Models
from cnn_model import cnn_model

# 60,000 images that are made up of 28x28 pixels
dataset = keras.datasets.fashion_mnist  # load dataset
(train, train_labels), (val, val_labels) = dataset.load_data()  # split into tetsing and training

# preprocess images
train = train.reshape(train.shape[0], 28, 28, 1)
val = val.reshape(val.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
train = train.astype('float32')
val = val.astype('float32')

# Divide rgb values bt 255, which results in values between 0 and 1
train = train / 255.0
val = val / 255.0

model = cnn_model(train, val, train_labels, val_labels, input_shape).AlexNet()

# evaluate model
test_loss, test_accuracy = model.trained_model.evaluate(val, val_labels, verbose=True)

print('Test accuracy:', test_accuracy)