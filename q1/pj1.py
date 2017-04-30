import csv

import numpy as np
from pyspark import SparkContext, SparkConf

import util
from util import data_path

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

train = sc.textFile(data_path + 'train.csv').filter(lambda line: line[0] != 'l')
train = train.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
train = train.map(lambda l: (map(int, l[-1].split(' ')), map(util.toFloat, l[:-1])))
train = train.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))
train = train.map(lambda l: (np.array(l[0]), np.array(l[1])))

# raw_valid_data = sc.textFile(data_path + 'valid.csv').filter(lambda line: line[0] != 'l')
# valid_data = raw_valid_data.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
# valid = valid_data.map(lambda l: (map(int, l[-1].split(' ')), map(util.toFloat, l[:-1])))
# valid = valid.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))

print 'train count\t' + str(train.count())
# print 'valid count\t' + str(valid.count())
# print train.sample(False, 0.003).collect()

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(256, input_dim=96 * 96))
model.add(Dropout(0.1))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=RMSprop())

from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers

spark_model = SparkModel(sc, model, frequency='epoch', mode='asynchronous', num_workers=1)
spark_model.train(train, nb_epoch=10, batch_size=8, verbose=1, validation_split=0.1)

del train

test_X, test_Y = util.load_np(data_path + 'valid.csv')
score = spark_model.master_network.evaluate(test_X, test_Y, batch_size=5)
print 'test loss = ' + str(score)

json_string = spark_model.master_network.to_json()
with open('model.json', 'w') as model:
    model.write(json_string)
spark_model.master_network.save_weights('model_weights.h5')






