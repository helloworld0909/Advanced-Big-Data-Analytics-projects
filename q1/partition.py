import random
import util
from util import data_path

def partition():
    proportion = 0.1
    with open('training.csv', 'r') as fin:
        lines = fin.readlines()
        train = open('train.csv', 'w')
        valid = open('valid.csv', 'w')
        random.shuffle(lines)
        train.writelines(lines[:int((1 - proportion) * len(lines))])
        valid.writelines(lines[int((1 - proportion) * len(lines)):])

        train.close()
        valid.close()

def plot():
    images = util.load_image(data_path + 'test.csv', normalize=False)
    imageId = random.randrange(len(images))
    image = images[imageId]
    with open('predict.txt', 'r') as predict:
        position = predict.readlines()[imageId].strip().split(' ')
        position = map(float, position)
        position = map(lambda x: x*48+48, position)
        print position
        util.visualize(image, position)

def predict():
    from keras.models import model_from_json
    model = model_from_json(open('model.json').read())
    model.load_weights('model_weights.h5')

    from keras.optimizers import Adam
    adam = Adam(lr=1e-4)
    model.compile(loss='mean_squared_error', optimizer=adam)

    test_X, test_Y = util.load_np(data_path + 'valid.csv')
    score = model.evaluate(test_X, test_Y, batch_size=5)
    print score

    test = util.load_image(data_path + 'test.csv')

    positions = model.predict(test, batch_size=1)
    print positions


if __name__ == '__main__':
    predict()



