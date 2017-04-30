import csv
import numpy as np
# from pyspark import SparkContext, SparkConf

import util
from util import data_path

# conf = SparkConf().setAppName('q1')
# sc = SparkContext(conf=conf)

# train = sc.textFile(data_path + 'train.csv').filter(lambda line: line[0] != 'l')
# train = train.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
# train = train.map(lambda l: (map(int, l[-1].split(' ')), map(util.toFloat, l[:-1])))
# train = train.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))
# train = train.map(lambda l: (np.array(l[0]), np.array(l[1])))
# train = train.map(lambda l: l[0].reshape((96, 96)))

train_X, train_Y = util.load_np(data_path + 'train.csv')

# train_count = train.count()
# print 'train count\t' + str(train_count)

print(len(train_X))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
# model.add(Dense(96 * 2, input_shape=(96 * 96, )))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30))

# Conv layer 1 output shape (32, 96, 96)
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',     # Padding method
    dim_ordering='th',      # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
    input_shape=(1,         # channels
                 96, 96,)    # height & width
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    # Padding method
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

# from elephas.spark_model import SparkModel
# from elephas import optimizers as elephas_optimizers
#
# adagrad = elephas_optimizers.Adagrad()
# spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=1)
# spark_model.train(train, nb_epoch=5, batch_size=4, verbose=1, validation_split=0.1)

model.fit(train_X, train_Y, epochs=1, verbose=1)

json_string = model.to_json()
with open('model.json', 'w') as model_out:
    model_out.write(json_string)
model.save_weights('model_weights.h5')

