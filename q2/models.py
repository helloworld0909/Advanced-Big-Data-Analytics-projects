from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy

nb_CNN_feature = 4096
nb_label = 239

def dense_CNN():
    model = Sequential()
    model.add(Dense(1024, input_shape=(None, nb_CNN_feature)))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_label))
    model.add(Activation('softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss=categorical_crossentropy)
    return model

