

import util
import models


STEP = 10
BATCH_SIZE = 32


def fit_generator():
    x_train = util.feature_generator('FCVID_CNN_sample_shuffle.txt')
    y_train = util.label_generator('FCVID_Label_sample_shuffle.txt')
    train = util.xy_generator(x_train, y_train)

    model = models.dense_CNN()

    model.fit_generator(train, steps_per_epoch=int(util.nb_sample/STEP), epochs=2, verbose=1)

    util.save_model(model)


def fit():
    x_train_CNN = util.load_data('FCVID_CNN_sample_shuffle.txt')
    x_train_IDT = util.load_data('FCVID_IDT_Traj_sample_shuffle.txt')
    x_train_MFCC = util.load_data('FCVID_MFCC_sample_shuffle.txt')
    y_train = util.load_data('FCVID_Label_sample_shuffle.txt')

    model = models.dense_fusion()
    model.fit([x_train_CNN, x_train_IDT, x_train_MFCC], y_train, epochs=20, verbose=1, batch_size=32, validation_split=0.1)

    util.save_model(model)

if __name__ == '__main__':
    fit()