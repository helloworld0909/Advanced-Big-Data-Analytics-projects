from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

def CNN_1():
    model = Sequential()
    # Conv layer 1 output shape (32, 96, 96)
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=5,
        nb_col=5,
        border_mode='same',  # Padding method
        dim_ordering='th',
        # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
        input_shape=(1,  # channels
                     96, 96,)  # height & width
    ))
    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same',  # Padding method
    ))

    # Conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))

    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(30))

    adam = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def CNN_2():
    model = Sequential()
    # Conv layer 1 output shape (32, 96, 96)
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=5,
        nb_col=5,
        border_mode='same',  # Padding method
        dim_ordering='th',
        # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
        input_shape=(1,  # channels
                     96, 96,)  # height & width
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same',  # Padding method
    ))

    model.add(Convolution2D(
        nb_filter=64,
        nb_row=2,
        nb_col=2,
        border_mode='same'
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        border_mode='same'
    ))

    model.add(Convolution2D(
        nb_filter=128,
        nb_row=2,
        nb_col=2,
        border_mode='same'
    ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(30))

    adam = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model