import csv
import numpy as np
# from pyspark import SparkContext, SparkConf

import util
from util import data_path
import model

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

model = model.CNN_2()

# from elephas.spark_model import SparkModel
# from elephas import optimizers as elephas_optimizers
#
# adagrad = elephas_optimizers.Adagrad()
# spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=1)
# spark_model.train(train, nb_epoch=5, batch_size=4, verbose=1, validation_split=0.1)

model.fit(train_X, train_Y, epochs=5, verbose=1)

json_string = model.to_json()
with open('model.json', 'w') as model_out:
    model_out.write(json_string)
model.save_weights('model_weights.h5')

