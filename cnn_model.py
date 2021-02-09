# import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization


class cnn_model:
    def __init__(self, train, val, train_labels, val_labels, input_shape):
        self.shape = input_shape
        self.train = train
        self.val = val
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.classes = 10
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        self.mc = ModelCheckpoint('best_model', monitor='val_accuracy', mode='max', save_best_only=True)

    '''
    This is just a simple model used for testing
    '''

    def cnn(self):
        # Build model
        model = keras.Sequential([
            Flatten(input_shape=self.shape),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax') 
        ])
        # compine and fit the model
        self.compile_and_fit(model)

    '''
    Build the AlexNet model. Currently the pooling layers reduce the input size too much
    this is why they are commented out. Waiting for TA's response to my mail
    '''

    def AlexNet(self):
        # Build the model
        model = keras.Sequential([
            Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu',
                   input_shape=self.shape, kernel_initializer='he_normal'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            # MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal'),
            # MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(1000, activation='relu'),
            Dense(10, activation='softmax'),
        ])

    
        self.compile_and_fit(model)

    '''
    Compile and fit the model, then return it. 
    '''

    def compile_and_fit(self, model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(self.train, self.train_labels, validation_data=(self.val, self.val_labels), epochs=1000,
                  callbacks=[self.es, self.mc])

        return model
