import numpy as np
import random

DATA_PATH = 'input/'
nb_sample = 91223


def sample_file(filename, step, start=0):
    count = 0
    out = open(DATA_PATH + filename.split('.')[0] + '_sample{}.txt'.format(str(start)), 'w')
    with open(DATA_PATH + filename, 'r') as input_file:
        for line in input_file:
            if count % step == start:
                out.write(line)
            count += 1
    out.close()


def shuffle_file(filename, indices):
    out = open(DATA_PATH + filename.split('.')[0] + '_shuffle.txt', 'w')
    with open(DATA_PATH + filename, 'r') as input_file:
        lines = input_file.readlines()
        length = len(lines)
        for index in indices:
            if index < length:
                out.write(lines[index])
    out.close()

def merge_file(filename, num):
    with open(DATA_PATH + filename.split('.')[0] + '_train.txt', 'w') as output_file:
        for index in range(1, num):
            fn = open(DATA_PATH + filename.split('.')[0] + '_sample{}'.format(index) + '_shuffle.txt', 'r')
            for line in fn:
                output_file.write(line)
            fn.close()

def item_generator(filename, buffer_size=4096):
    with open(DATA_PATH + filename, 'r') as fn:
        buffer = []
        cnt = 0
        for line in fn:
            tmp = list(map(float, line.strip().split('\t')))
            buffer.append(np.array(tmp).reshape(len(tmp)))
            cnt += 1
            if cnt >= buffer_size:
                for item in buffer:
                    yield np.array(item)
                buffer = []
                cnt = 0

def xy_generator(x1_generator, x2_generator, x3_generator, y_generator, batch_size=32):
    x1_batch = []
    x2_batch = []
    x3_batch = []
    y_batch = []
    while True:
        for _ in range(batch_size):
            x1 = next(x1_generator)
            x2 = next(x2_generator)
            x3 = next(x3_generator)
            y = next(y_generator)
            x1_batch.append(x1)
            x2_batch.append(x2)
            x3_batch.append(x3)
            y_batch.append(y)
        yield [np.array(x1_batch), np.array(x2_batch), np.array(x3_batch)], np.array(y_batch)
        x1_batch = []
        x2_batch = []
        x3_batch = []
        y_batch = []


def load_data(filename):
    with open(DATA_PATH + filename, 'r') as fn:
        data_list = []
        for line in fn:
            data_list.append(tuple(map(float, line.strip().split('\t'))))
        return np.array(data_list)

def save_model(model):
    model_json = model.to_json()
    with open('model.json', 'w') as model_out:
        model_out.write(model_json)
    model.save_weights('model_weights.h5')

if __name__ == '__main__':
    merge_file('FCVID_CNN.txt', 10)
    merge_file('FCVID_IDT_Traj.txt', 10)
    merge_file('FCVID_MFCC.txt', 10)
    merge_file('FCVID_Label.txt', 10)


