import numpy as np
import random

DATA_PATH = 'input/'
nb_sample = 91223

def shuffle_file(x_file, y_file):
    x_fn = open(DATA_PATH + x_file, 'r')
    y_fn = open(DATA_PATH + y_file, 'r')
    fn = zip(x_fn, y_fn)
    random.shuffle(fn)

    x_shuffle = open(x_file + '_shuffle', 'w')
    y_shuffle = open(y_file + '_shuffle', 'w')
    for x_line, y_line in fn:
        x_shuffle.write(x_line)
        y_shuffle.write(y_line)

def feature_generator(filename):
    with open(DATA_PATH + filename, 'r') as fn:
        for line in fn:
            yield np.array(list(map(float, line.strip().split('\t'))))

def label_generator(filename):
    with open(DATA_PATH + filename, 'r') as fn:
        for line in fn:
            yield np.array(list(map(float, line.strip().split('\t'))))


def xy_generator(x_generator, y_generator, batch_size=1):
    batch = []
    for item in zip(x_generator, y_generator):
        batch.append(item)
        if len(batch) == batch_size:


            yield batch
            batch = []

if __name__ == '__main__':
    x_train = feature_generator('FCVID_CNN.txt')
    y_train = label_generator('FCVID_Label.txt')
    train = xy_generator(x_train, y_train, batch_size=2)
    print(next(train))
