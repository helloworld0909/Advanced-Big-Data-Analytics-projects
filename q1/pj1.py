import random

import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext

conf = SparkConf().setAppName('q1')
sc = SparkContext(conf=conf)

data_path = ''

# fn = open('training.csv', 'r')
# trainFile = fn.readlines()
# attrName = trainFile[0].strip().split(',')
# trainData = []
# for line in trainFile[1:100]:
#     labels = line.strip().split(',')[:-1]
#     labelTuple = []
#     image = map(int, line.strip().split(',')[-1].split(' '))
#     for label in labels:
#         try:
#             labelTuple.append(float(label))
#         except:
#             labelTuple.append(None)
#     trainData.append(labelTuple.append(image))
#
# trainRDD = sc.parallelize(trainData)
# print trainRDD.collect()

sqlContext = SQLContext(sc)


def shuffle_csv(csv_file):
    lines = open(csv_file).readlines()
    random.shuffle(lines)
    open(csv_file, 'w').writelines(lines)


def load_data_frame(csv_file, shuffle=True):
    if shuffle:
        shuffle_csv(csv_file)
    data = sc.textFile(data_path + csv_file)  # This is an RDD, which will later be transformed to a data frame
    data = data.filter(lambda x: x[0].isdigit()).map(lambda line: line.split(','))

    def convertData(line):
        image = map(int, line[-1].split(' '))
        features = []
        for feature in line[:-1]:
            try:
                features.append(float(feature))
            except:
                features.append(None)
        return Vectors.dense(np.asarray(features).astype(np.float64)), image

    data = data.map(convertData)
    return sqlContext.createDataFrame(data, ['features', 'image'])

train_df = load_data_frame("training.csv")
print("Train data frame:")
train_df.show(5)