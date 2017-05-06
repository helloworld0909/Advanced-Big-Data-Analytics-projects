from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2

nb_CNN_feature = 4096
nb_IDT_feature = 4000
nb_MFCC_feature = 4000
nb_label = 239


def dense_CNN():
    model = Sequential()
    model.add(Dense(1024, input_shape=(nb_CNN_feature,)))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_label, kernel_regularizer=l2(0.02)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])
    return model


def dense_fusion():
    DROPOUT_RATE = 0.5

    cnn_input = Input(shape=(nb_CNN_feature,), name='cnn_input')
    cnn = Dense(1024, input_shape=(nb_CNN_feature,), activation='relu')(cnn_input)
    cnn = Dropout(DROPOUT_RATE)(cnn)
    cnn = Dense(256, activation='relu')(cnn)

    idt_input = Input(shape=(nb_IDT_feature,), name='idt_input')
    idt = Dense(1024, input_shape=(nb_IDT_feature,), activation='relu')(idt_input)
    idt = Dropout(DROPOUT_RATE)(idt)
    idt = Dense(256, activation='relu')(idt)

    mfcc_input = Input(shape=(nb_MFCC_feature,), name='mfcc_input')
    mfcc = Dense(1024, input_shape=(nb_MFCC_feature,), activation='relu')(mfcc_input)
    mfcc = Dropout(DROPOUT_RATE)(mfcc)
    mfcc = Dense(256, activation='relu')(mfcc)

    fusion = concatenate([cnn, idt, mfcc])
    fusion = Dropout(DROPOUT_RATE)(fusion)
    fusion = Dense(nb_label, activation='softmax', kernel_regularizer=l2(0.02))(fusion)

    model = Model(inputs=[cnn_input, idt_input, mfcc_input], outputs=fusion)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

    return model

def compile_model(model):
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

