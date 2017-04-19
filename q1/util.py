import numpy as np

def toFloat(num):
    try:
        return (float(num) - 48) / 48
    except:
        return None

def load_np(filename):
    fn = open(filename, 'r')
    testFile = fn.readlines()
    testData_X = []
    testData_Y = []
    for line in testFile:
        labels = line.strip().split(',')[:-1]
        if '' not in labels and '0' <= line[0] <= '9':
            labelTuple = []
            image = map(int, line.strip().split(',')[-1].split(' '))
            image = map(lambda pixel: pixel / 255.0, image)
            for label in labels:
                try:
                    labelTuple.append((float(label) - 48) / 48)
                except:
                    labelTuple.append(None)
            testData_X.append(image)
            testData_Y.append(labelTuple)
    X = np.array(testData_X)
    Y = np.array(testData_Y)
    fn.close()
    return X, Y

def load_image(filename):
    images = []
    fn = open(filename, 'r')
    for line in fn:
        if '0' <= line[0] <= '9':
            num, image = line.strip().split(',')
            image = image.split(' ')
            images.append((num, image))
    return images

