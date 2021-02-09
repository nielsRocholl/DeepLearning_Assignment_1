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

train.shape

# 9 items of clothging, 9 classes
classes = ['top/T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Devide rgb values bt 255, which results in values between 0 and 1
train = train / 255.0

val = val / 255.0

model = cnn_model(train, val, train_labels, val_labels).cnn()

# evaluate model

test_loss, test_accuracy = model.evaluate(val, val_labels, verbose=True)

print('Test accuracy:', test_accuracy)


predict = model.predict(val)

predict[0]

np.argmax(predict[0])

val_labels[0]

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()
