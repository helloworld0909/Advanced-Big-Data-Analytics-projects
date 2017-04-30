import numpy as np
try:
    from matplotlib import pyplot
except:
    pass

data_path = 'data/'

featureNames = {
    'left_eye_center_x': 0,
    'left_eye_center_y': 1,
    'left_eye_inner_corner_x': 4,
    'left_eye_inner_corner_y': 5,
    'left_eye_outer_corner_x': 6,
    'left_eye_outer_corner_y': 7,
    'left_eyebrow_inner_end_x': 12,
    'left_eyebrow_inner_end_y': 13,
    'left_eyebrow_outer_end_x': 14,
    'left_eyebrow_outer_end_y': 15,
    'mouth_center_bottom_lip_x': 28,
    'mouth_center_bottom_lip_y': 29,
    'mouth_center_top_lip_x': 26,
    'mouth_center_top_lip_y': 27,
    'mouth_left_corner_x': 22,
    'mouth_left_corner_y': 23,
    'mouth_right_corner_x': 24,
    'mouth_right_corner_y': 25,
    'nose_tip_x': 20,
    'nose_tip_y': 21,
    'right_eye_center_x': 2,
    'right_eye_center_y': 3,
    'right_eye_inner_corner_x': 8,
    'right_eye_inner_corner_y': 9,
    'right_eye_outer_corner_x': 10,
    'right_eye_outer_corner_y': 11,
    'right_eyebrow_inner_end_x': 16,
    'right_eyebrow_inner_end_y': 17,
    'right_eyebrow_outer_end_x': 18,
    'right_eyebrow_outer_end_y': 19
}

def toFloat(num):
    try:
        return float(num) / 96
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
                labelTuple.append(toFloat(label))
            testData_X.append(image)
            testData_Y.append(labelTuple)
    X = np.array(testData_X)
    Y = np.array(testData_Y)
    fn.close()
    return X, Y

def load_image(filename, normalize=True):
    images = []
    fn = open(filename, 'r')
    for line in fn:
        if '0' <= line[0] <= '9':
            num, image = line.strip().split(',')
            image = map(int, image.strip().split(' '))
            if normalize:
                image = map(lambda pixel: pixel / 255.0, image)
            images.append(image)
    fn.close()
    return np.array(images)

def visualize(image, position):
    img = image.reshape((96, 96))
    imgPlot = pyplot.imshow(img, cmap='gray')
    pyplot.scatter(position[0::2], position[1::2], marker='x', s=10)
    pyplot.show()


