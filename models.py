# ------------------------------------------------------------------------------
#  Comparison of regularization techniqes.
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Implementation of different regularized models
# ------------------------------------------------------------------------------
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from keras import backend as K

def get_available_models():
    return ['baseline',
            'batchnorm',
            'weightnorm',
            'fastdropout']

def get_model(model_name, data):
    assert model_name in get_available_models(), 'Invalid model name'

    if model_name == 'baseline':
        return get_baseline_model(data)
    if model_name == 'batchnorm':
        return get_batchnorm_model(data)
    if model_name == 'weightnorm':
        return get_weightnorm_model(data)
    if model_name == 'fastdropout':
        return get_fast_dropout_model(data)

def get_baseline_model(data, use_dropout=False, dropout_ratio=0.2):
    num_classes = data[0]
    img_size = data[1].shape[1]
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(img_size,)))
    if use_dropout:
        model.add(Dropout(dropout_ratio))
    model.add(Dense(512, activation='relu'))
    if use_dropout:
        model.add(Dropout(dropout_ratio))
    model.add(Dense(num_classes, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

def get_batchnorm_model(data):
    num_classes = data[0]
    img_size = data[1].shape[1]
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(img_size,)))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model

def get_weightnorm_model(data):
    from weightnorm import SGDWithWeightnorm
    from weightnorm import data_based_init

    num_classes = data[0]
    img_size = data[1].shape[1]
    x_train = data[1]

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(img_size,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    #model.summary()
    sgd_wn = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd_wn,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    data_based_init(model, x_train[:100])
    return model

def get_fast_dropout_model(data,
                           use_dropout_cross_entropy=False,
                           use_variance_propagation=True,
                           dropout_ratio=0.5):
    from fast_dropout import FastDropoutDenseLayer
    num_classes = data[0]
    img_size = data[1].shape[1]
    model = Sequential()
    model.add(FastDropoutDenseLayer(512, input_shape=(img_size,),
                                    has_var_input=False,
                                    has_var_output=use_variance_propagation,
                                    dropout=dropout_ratio))
    model.add(FastDropoutDenseLayer(512, has_var_output=False,
                                    has_var_input=use_variance_propagation,
                                    dropout=dropout_ratio))
    if use_dropout_cross_entropy:
        model.add(FastDropoutDenseLayer(num_classes, activation='linear',
                                        dropout=dropout_ratio))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    #model.summary()

    if use_dropout_cross_entropy:
        from fast_dropout import FastDropoutCrossEntropyLoss
        loss = FastDropoutCrossEntropyLoss
    else:
        loss = 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=RMSprop(),
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
