import util
import models

STEP = 50
BATCH_SIZE = 32


def test():
    x_train = util.feature_generator('FCVID_CNN.txt')
    y_train = util.label_generator('FCVID_Label.txt')
    train = util.xy_generator(x_train, y_train, batch_size=BATCH_SIZE)

    model = models.dense_CNN()


    model.fit_generator(train, steps_per_epoch=int(util.nb_sample/BATCH_SIZE), epochs=5, verbose=1)


if __name__ == '__main__':
    test()