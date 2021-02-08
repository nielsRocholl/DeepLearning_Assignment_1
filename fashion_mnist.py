import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 60,000 images that are made up of 28x28 pixels
dataset = keras.datasets.fashion_mnist  # load dataset

(train, train_labels), (test, test_labels) = dataset.load_data()  # split into tetsing and training

train.shape

# 9 items of clothging, 9 classes
classes = ['top/T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Devide rgb values bt 255, which results in values between 0 and 1
train = train / 255.0

test = test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])


# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train model
model.fit(train, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!

# evaluate model

test_loss, test_accuracy = model.evaluate(test, test_labels, verbose=True)

print('Test accuracy:', test_accuracy)


predict = model.predict(test)

predict[0]

np.argmax(predict[0])

test_labels[0]

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
