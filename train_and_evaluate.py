import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Layer, Activation
from keras.models import Sequential
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Pad images with 0s
train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

epochs = 10


class CustomBatchNormalization(Layer):
    def __init__(self, scale, **kwargs):
        self.use_scale = scale
        super(CustomBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[-1],)
        if self.use_scale is True:
            self.scale = self.add_weight(name='scale',
                                         shape=shape,
                                         initializer='ones',
                                         trainable=True)

        self.shift = self.add_weight(name='shift',
                                     shape=shape,
                                     initializer='zeros',
                                     trainable=True)

        super(CustomBatchNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x)
        variance = K.var(x)

        x -= mean
        x /= K.sqrt(variance) + 10 ** -3

        if self.use_scale is True:
            x = self.scale * x + self.shift
        else:
            x += self.shift
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def build_model_without_bn():
    # LeNet-5
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_model_with_bn():
    model = keras.Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), use_bias=False, input_shape=(32, 32, 1)))
    model.add(CustomBatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), use_bias=False))
    model.add(CustomBatchNormalization(scale=False))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, use_bias=False))
    model.add(CustomBatchNormalization(scale=False))

    model.add(Activation('relu'))

    model.add(Dense(units=84, use_bias=False))
    model.add(CustomBatchNormalization(scale=False))

    model.add(Activation('relu'))

    model.add(Dense(units=10, use_bias=False))
    model.add(CustomBatchNormalization(scale=True))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model_without_bn = build_model_without_bn()
history_without_bn = model_without_bn.fit(train_images, train_labels, epochs=epochs, batch_size=64)


model_with_bn = build_model_with_bn()
history_with_bn = model_with_bn.fit(train_images, train_labels, epochs=epochs, batch_size=64)

plt.plot(history_without_bn.history['acc'], label='accuracy without BN')
plt.plot(history_with_bn.history['acc'], label='accuracy with BN')

plt.title('accuracy')
plt.legend()
plt.show()
