import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint


class cnn_model:
    def __init__(self, train, val, train_labels, val_labels):
        self.train = train
        self.val = val
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.optimizer = 'adam'
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        self.mc = ModelCheckpoint('best_model', monitor='val_accuracy', mode='max', save_best_only=True)

    def cnn(self):
        # Build model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
            keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
            keras.layers.Dense(10, activation='softmax')  # output layer (3)
        ])

        # compile model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.train, self.train_labels, validation_data=(self.val, self.val_labels), epochs=1000,
                  callbacks=[self.es, self.mc])
        return model
