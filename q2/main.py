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
    x_train_CNN = util.load_data('FCVID_CNN_sample_shuffle.txt')
    x_train_IDT = util.load_data('FCVID_IDT_Traj_sample_shuffle.txt')
    x_train_MFCC = util.load_data('FCVID_MFCC_sample_shuffle.txt')
    y_train = util.load_data('FCVID_Label_sample_shuffle.txt')

    model = model_from_json(open('models/model.json').read())
    model.load_weights('models/model_weights.h5')
    models.compile_model(model)

    model.fit([x_train_CNN, x_train_IDT, x_train_MFCC], y_train, epochs=150, verbose=1, batch_size=32, validation_split=0.1)

    util.save_model(model)

if __name__ == '__main__':
    fit()
