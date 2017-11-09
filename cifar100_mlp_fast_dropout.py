#!/usr/bin/env python
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import random as rn
np.random.seed(25)
rn.seed(7448)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf

import keras
from keras.datasets import mnist, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from fast_dropout_layer import FastDropoutDenseLayer, FastDropoutCrossEntropyLoss


session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


batch_size = 128
num_classes = 100
epochs = 2

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

img_size = 32*32*3
x_train = x_train.reshape(-1, img_size)
x_test = x_test.reshape(-1, img_size)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

model = Sequential()
model.add(FastDropoutDenseLayer(512, input_shape=(img_size,), has_var_input=False))
model.add(FastDropoutDenseLayer(512, has_var_output=False))
#model.add(FastDropoutDenseLayer(num_classes, activation='linear'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',#FastDropoutCrossEntropyLoss,#
              optimizer=RMSprop(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
