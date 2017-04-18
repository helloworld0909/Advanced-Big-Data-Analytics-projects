import numpy as np
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

data_path = ''


def toFloat(num):
    try:
        return (float(num) - 48) / 48
    except:
        return None

raw_data = sc.textFile(data_path + 'train.csv').filter(lambda line: line[0] != 'l')
train_data = raw_data.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
train = train_data.map(lambda l: (map(int, l[-1].split(' ')), map(toFloat, l[:-1])))
train = train.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))

raw_valid_data = sc.textFile(data_path + 'valid.csv').filter(lambda line: line[0] != 'l')
valid_data = raw_valid_data.map(lambda line: line.strip().split(',')).filter(lambda l: '' not in l)
valid = valid_data.map(lambda l: (map(int, l[-1].split(' ')), map(toFloat, l[:-1])))
valid = valid.map(lambda l: (map(lambda pixel: pixel / 255.0, l[0]), l[1]))

print 'train count\t' + str(train.count())
print 'valid count\t' + str(valid.count())

def load_np(filename):
    fn = open(filename, 'r')
    testFile = fn.readlines()
    testData_X = []
    testData_Y = []
    for line in testFile:
        labels = line.strip().split(',')[:-1]
        if '' not in labels and line[0] != 'l':
            labelTuple = []
            image = map(int, line.strip().split(',')[-1].split(' '))
            image = map(lambda pixel: pixel / 255.0, image)
            for label in labels:
                try:
                    labelTuple.append((float(label) - 48) / 48)
                except:
                    labelTuple.append(None)
            testData_X.append(image)
            testData_Y.append(labelTuple)
    X = np.array(testData_X)
    Y = np.array(testData_Y)
    fn.close()
    return X, Y

test_X, test_Y = load_np('valid.csv')

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
print score
