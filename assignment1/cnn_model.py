import os
import numpy as np
import pickle
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Model
from keras.applications import InceptionV3, ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential, load_model



'''
A class that contains CNN architectures. This class can be used to create and train a
CNN model. Once it is trained the model is returned. 
'''


class cnn_model:
    def __init__(self, steps_per_epoch, validation_steps, train, val, input_shape, model, activation='relu', optimizer='adam', epochs = 'es'):
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.optimizer = optimizer
        self.activation = activation
        self.shape = input_shape
        self.train = train
        self.val = val
        self.classes = 3
        self.model = model
        if epochs == 'es':
            self.es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
            self.epochs = 100
        else:
            self.es = None
            self.epochs = epochs
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
        base_model = InceptionV3    (include_top=False, weights=None, input_shape=self.shape)
        x = base_model.output
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        self.compile_and_fit(model)

    '''
    Build ResNet. Keras has a premade model in "applications", however we do not load the weights. 
    '''

    def ResNet(self):
        base_model = ResNet50(weights=None, include_top=True, input_shape=self.shape, classes = 3)
        x = base_model.output
        model = Model(inputs=base_model.input, outputs=x)

        self.compile_and_fit(model)


    def LeNet(self):
        model = Sequential([
          Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=self.shape),
          AveragePooling2D(),
          Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
          AveragePooling2D(),
          Flatten(),
          Dense(units=120, activation='relu'),
          Dense(units=84, activation='relu'),
          Dense(units=self.classes, activation = 'softmax'),
        ])

        self.compile_and_fit(model)



    def compile_and_fit(self, model):
        """Compile and train the model"""
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if self.es is None:
            callbacks = [self.mc]
        else:
            callbacks = [self.es, self.mc]
        final_model = model.fit(
            x=self.train.repeat(),
            validation_data=self.val.repeat(),
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Load best model checkpoint
        self.model = load_model(self.mc.filepath)
        self.final_model = final_model
        self.calculate_confusion()

    def calculate_confusion(self):
        predictions = predictions = self.model.predict(self.val)
        validation_classes = np.concatenate([y for (x,y) in self.val.as_numpy_iterator()])
        predicted_classes = np.argmax(predictions, axis=1)
        self.confusion_matrix = np.zeros((self.classes, self.classes))
        for predicted in range(self.classes):
            for actual in range(self.classes):
                self.confusion_matrix[predicted, actual] = np.sum(np.logical_and(
                    predicted_classes == predicted,
                    validation_classes == actual
                ))
        
    def save_final_model(self, output_dir):
        model_path = os.path.join(output_dir, "model.h5")
        history_path = os.path.join(output_dir, "history.npy")
        confusion_path = os.path.join(output_dir, "confusion.npy")        
        self.model.save(model_path)
        np.save(history_path, self.final_model.history)
        np.save(confusion_path, self.confusion_matrix)
   
