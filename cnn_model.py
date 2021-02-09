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

'''
A class that contains CNN architectures. This class can be used to create and train a
CNN model. Once it is trained the model is returned. 
'''

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
        model = Sequential([
            Flatten(input_shape=self.shape),
            Dense(128, activation='relu'),
            Dense(self.classes, activation='softmax') 
        ])
        # compine and fit the model
        self.compile_and_fit(model)

    '''
    Build the AlexNet model. Currently the pooling layers reduce the input size too much
    this is why they are commented out. Waiting for TA's response to my mail
    '''

    def AlexNet(self):
        # Build the model
        model = Sequential([
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
            Dense(self.classes, activation='softmax'),
        ])

    
        self.compile_and_fit(model)



    '''
    Build the VGG model. Same pooling layer problem as with AlexNet
    '''

    def VGG(self):
        model = Sequential([
            Conv2D(input_shape=self.shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Flatten(),
            Dense(units=4096,activation="relu"),
            Dense(units=4096,activation="relu"),
            Dense(units=self.classes, activation="softmax"),
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
