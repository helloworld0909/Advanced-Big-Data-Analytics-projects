import numpy as np
import random

DATA_PATH = 'input/'
nb_sample = 91223


def sample_file(filename, step):
    count = 0
    out = open(DATA_PATH + filename.split('.')[0] + '_sample.txt', 'w')
    with open(DATA_PATH + filename, 'r') as input_file:
        for line in input_file:
            if count % step == 0:
                out.write(line)
            count += 1
    out.close()


def shuffle_file(filename, indices):
    out = open(DATA_PATH + filename.split('.')[0] + '_shuffle.txt', 'w')
    with open(DATA_PATH + filename, 'r') as input_file:
        lines = input_file.readlines()
        for index in indices:
            out.write(lines[index])
    out.close()

def feature_generator(filename):
    with open(DATA_PATH + filename, 'r') as fn:
        for line in fn:
            yield np.array(list(map(float, line.strip().split('\t')))).reshape(1, 4096)

def label_generator(filename):
    with open(DATA_PATH + filename, 'r') as fn:
        for line in fn:
            yield np.array(list(map(float, line.strip().split('\t')))).reshape(1, 239)


def xy_generator(x_generator, y_generator):
    return zip(x_generator, y_generator)

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
    indices = list(range(9123))
    random.shuffle(indices)
    print(indices)
    shuffle_file('FCVID_CNN_sample.txt', indices)
    shuffle_file('FCVID_IDT_Traj_sample.txt', indices)
    shuffle_file('FCVID_MFCC_sample.txt', indices)
    shuffle_file('FCVID_Label_sample.txt', indices)
    pass
