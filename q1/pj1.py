from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

data_path = ''

# attrName = trainFile[0].strip().split(',')

def toFloat(num):
    try:
        return float(num)
    except:
        return None

raw_data = sc.textFile(data_path + 'training.csv').filter(lambda line: line[0] != 'l')
train_data = raw_data.map(lambda line: line.strip().split(','))
train_X = train_data.map(lambda l: map(int, l[-1].split(' ')))
train_Y = train_data.map(lambda l: l[:-1]).map(lambda l: map(toFloat, l))

print 'train count\t' + str(train_X.count())




# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
#
# model = Sequential()
# model.add(Dense(128, input_dim=96*96))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30))
# model.compile(loss='categorical_crossentropy', optimizer=SGD())
#
# from elephas.spark_model import SparkModel
# from elephas import optimizers as elephas_optimizers
#
# adagrad = elephas_optimizers.Adagrad()
# spark_model = SparkModel(sc, model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=1)
# spark_model.train(trainRDD, nb_epoch=20, batch_size=trainRDD.count(), verbose=0, validation_split=0.1)