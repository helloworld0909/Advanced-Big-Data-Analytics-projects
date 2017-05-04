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

def item_generator(filename, buffer_size=1024):
    with open(DATA_PATH + filename, 'r') as fn:
        buffer = []
        cnt = 0
        for line in fn:
            tmp = list(map(float, line.strip().split('\t')))
            buffer.append(np.array(tmp).reshape(1, len(tmp)))
            cnt += 1
            if cnt >= buffer_size:
                for item in buffer:
                    yield np.array(item)
                buffer = []
                cnt = 0

def xy_generator(x_generator, y_generator, batch_size=32):
    x_batch = []
    y_batch = []
    generator = zip(x_generator, y_generator)
    while True:
        for _ in range(batch_size):
            x, y = next(generator)
            x_batch.append(x)
            y_batch.append(y)
        yield np.array(x_batch), np.array(y_batch)


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
    for i in range(1,10):
        indices = list(range(9123))
        random.shuffle(indices)
        print(i)
        sample_file('FCVID_CNN.txt', 10, start=i)
        sample_file('FCVID_IDT_Traj.txt', 10, start=i)
        sample_file('FCVID_MFCC.txt', 10, start=i)
        sample_file('FCVID_Label.txt', 10, start=i)

        shuffle_file('FCVID_CNN_sample{}.txt'.format(str(i)), indices)
        shuffle_file('FCVID_IDT_Traj_sample{}.txt'.format(str(i)), indices)
        shuffle_file('FCVID_MFCC_sample{}.txt'.format(str(i)), indices)
        shuffle_file('FCVID_Label_sample{}.txt'.format(str(i)), indices)

