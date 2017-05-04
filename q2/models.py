from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.regularizers import l2

nb_CNN_feature = 4096
nb_label = 239

def dense_CNN():
    model = Sequential()
    model.add(Dense(1024, input_shape=(nb_CNN_feature, )))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_label, kernel_regularizer=l2(0.02)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
    return model

