from keras import Model
from keras.applications import InceptionV3, ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

import matplotlib.pyplot as plt

'''
A class that contains CNN architectures. This class can be used to create and train a
CNN model. Once it is trained the model is returned. 
'''


class cnn_model:
    def __init__(self, steps_per_epoch, validation_steps, train, val, input_shape, activation='relu', optimizer='adam'):
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.optimizer = optimizer
        self.activation = activation
        self.shape = input_shape
        self.train = train
        self.val = val
        self.classes = 3
        self.es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        self.mc = ModelCheckpoint('best_model', monitor='val_accuracy', mode='max', save_best_only=True)

    '''
    This is just a simple model used for testing
    '''

    def cnn(self):
        # Build model
        model = Sequential([
            Flatten(input_shape=self.shape),
            Dense(128, activation=self.activation),
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
            Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation=self.activation,
                   input_shape=self.shape, kernel_initializer='he_normal'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation=self.activation,
                   kernel_initializer='he_normal'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation=self.activation,
                   kernel_initializer='he_normal'),
            Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation=self.activation,
                   kernel_initializer='he_normal'),
            Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=self.activation,
                   kernel_initializer='he_normal'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format=None),
            Flatten(),
            Dense(4096, activation=self.activation),
            Dense(4096, activation=self.activation),
            Dense(1000, activation=self.activation),
            Dense(self.classes, activation='softmax'),
        ])

        self.compile_and_fit(model)

    '''
    Build the VGG model.
    '''

    def VGG(self):
        model = Sequential([
            Conv2D(input_shape=self.shape, filters=64, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=self.activation),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(units=4096, activation=self.activation),
            Dense(units=4096, activation=self.activation),
            Dense(units=self.classes, activation="softmax"),
        ])

        self.compile_and_fit(model)

    '''
    TODO: Input shape needs to be at least 75x75
    '''

    def InceptionV3(self):
        base_model = InceptionV3(include_top=False, weights=None, input_shape=self.shape)
        x = base_model.output
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        self.compile_and_fit(model)

    '''
    Build ResNet. Keras has a premade model in "applications", however we do not load the weights. 
    '''

    def ResNet(self):
        base_model = ResNet50(weights=None, include_top=False, input_shape=self.shape)
        x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        # x = Dropout(0.7)(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        self.compile_and_fit(model)

    '''
    Compile and fit the model, then return it. 
    '''

    def compile_and_fit(self, model):
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        final_model = model.fit(
            x=self.train.repeat(),
            validation_data=self.val.repeat(),
            epochs=15,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=[self.es, self.mc],
            verbose=1
        )

        self.plot_training_results(final_model)

        return final_model

    def plot_training_results(self, final_model):
        accuracy = final_model.history['accuracy']
        val_accuracy = final_model.history['val_accuracy']

        loss = final_model.history['loss']
        val_loss = final_model.history['val_loss']


        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(accuracy, label='Training set')
        plt.plot(val_accuracy, label='Test set', linestyle='--')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)

        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training set')
        plt.plot(val_loss, label='Test set', linestyle='--')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)

        plt.show()
