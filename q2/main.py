import json
from keras.models import model_from_json

import util
import models


STEP = 10
BATCH_SIZE = 32


def fit_generator():
    x_train_CNN = util.item_generator('FCVID_CNN_train.txt')
    x_train_IDT = util.item_generator('FCVID_IDT_Traj_train.txt')
    x_train_MFCC = util.item_generator('FCVID_MFCC_train.txt')
    y_train = util.item_generator('FCVID_Label_train.txt')
    train = util.xy_generator(x_train_CNN, x_train_IDT, x_train_MFCC, y_train)

    model = model_from_json(open('models/model.json').read())
    model.load_weights('models/model_weights.h5')
    models.compile_model(model)

    model.fit_generator(train, steps_per_epoch=int(util.nb_sample*0.9/BATCH_SIZE), epochs=1, verbose=1)

    x_test_CNN = util.load_data('FCVID_CNN_sample_shuffle.txt')
    x_test_IDT = util.load_data('FCVID_IDT_Traj_sample_shuffle.txt')
    x_test_MFCC = util.load_data('FCVID_MFCC_sample_shuffle.txt')
    y_test = util.load_data('FCVID_Label_sample_shuffle.txt')

    score = model.evaluate([x_test_CNN, x_test_IDT, x_test_MFCC], y_test, verbose=1)
    print(score)

    util.save_model(model)


def fit():
    sample_num = 4
    assert sample_num < 9

    x_train_CNN = util.load_data('FCVID_CNN_sample{}_shuffle.txt'.format(sample_num))
    x_train_IDT = util.load_data('FCVID_IDT_Traj_sample{}_shuffle.txt'.format(sample_num))
    x_train_MFCC = util.load_data('FCVID_MFCC_sample{}_shuffle.txt'.format(sample_num))
    y_train = util.load_data('FCVID_Label_sample{}_shuffle.txt'.format(sample_num))

    x_test_CNN = util.load_data('FCVID_CNN_sample9_shuffle.txt')
    x_test_IDT = util.load_data('FCVID_IDT_Traj_sample9_shuffle.txt')
    x_test_MFCC = util.load_data('FCVID_MFCC_sample9_shuffle.txt')
    y_test = util.load_data('FCVID_Label_sample9_shuffle.txt')

    # model = model_from_json(open('models/model.json').read())
    model = models.dense_fusion()
    model.load_weights('models/model_weights.h5')
    models.compile_model(model)

    history = model.fit([x_train_CNN, x_train_IDT, x_train_MFCC], y_train, epochs=10, verbose=1, batch_size=128, validation_data=([x_test_CNN, x_test_IDT, x_test_MFCC], y_test))

    util.save_model(model)

    with open('history.txt', 'w') as hist_file:
        history_string = json.dumps(history.history)
        hist_file.write(history_string)


if __name__ == '__main__':
    fit()
