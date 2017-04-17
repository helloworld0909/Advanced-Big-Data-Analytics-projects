import random

import numpy as np
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

data_path = ''

fn = open('training.csv', 'r')
trainFile = fn.readlines()
attrName = trainFile[0].strip().split(',')
trainData = []
for line in trainFile[1:100]:
    labels = line.strip().split(',')[:-1]
    labelTuple = []
    image = map(int, line.strip().split(',')[-1].split(' '))
    for label in labels:
        try:
            labelTuple.append(float(label))
        except:
            labelTuple.append(None)
    trainData.append([labelTuple, image])

trainRDD = sc.parallelize(trainData)
print(trainRDD.count())

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from elephas.utils.rdd_utils import to_simple_rdd

model = Sequential()
model.add(Dense(128, input_dim=96*96))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(30))
model.compile(loss='categorical_crossentropy', optimizer=SGD())