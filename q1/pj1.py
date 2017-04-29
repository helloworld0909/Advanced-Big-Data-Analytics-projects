import csv

import numpy as np
from pyspark import SparkContext, SparkConf

import util
from util import data_path

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

raw_data = sc.textFile(data_path + 'train.csv').filter(lambda line: line[0] != 'l')
train_data = raw_data.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
train = train_data.map(lambda l: (map(int, l[-1].split(' ')), map(util.toFloat, l[:-1])))
train = train.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))

# raw_valid_data = sc.textFile(data_path + 'valid.csv').filter(lambda line: line[0] != 'l')
# valid_data = raw_valid_data.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
# valid = valid_data.map(lambda l: (map(int, l[-1].split(' ')), map(util.toFloat, l[:-1])))
# valid = valid.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))

print 'train count\t' + str(train.count())
# print 'valid count\t' + str(valid.count())

test_X, test_Y = util.load_np(data_path + 'valid.csv')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(128, input_dim=96 * 96))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(30))

model.compile(loss='mean_squared_error', optimizer=RMSprop())

from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers

adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=1)
spark_model.train(train, nb_epoch=5, batch_size=5, verbose=1, validation_split=0.1)

score = spark_model.master_network.evaluate(test_X, test_Y, batch_size=5)
print 'test score = ' + str(score)

# json_string = spark_model.master_network.to_json()
# with open('model.json', 'w') as model:
#     model.write(json_string)
# spark_model.master_network.save_weights('model_weights.h5')

test = util.load_image(data_path + 'test.csv')
positions = spark_model.master_network.predict_on_batch(test[0])

np.savetxt('predict.txt', positions)

# with open(data_path + 'IdLookupTable.csv') as id_file:
#     with open('output.csv', 'w') as output:
#         id_lookup = csv.reader(id_file)
#         labels = id_lookup.next()  # Skip the first row
#         output.write('RowId,Location\n')
#
#         for each in id_lookup:
#             rowId, imageId, featureName = each
#             p = positions[int(imageId) - 1][util.featureNames[featureName]]
#             position = float(p) * 48 + 48
#             output.write(str(rowId) + ',' + str(position) + '\n')

