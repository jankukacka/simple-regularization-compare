#!/usr/bin/env python
# ------------------------------------------------------------------------------
#  Comparison of regularization techniqes.
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Simple comparison of the following regularization techniqes:
#    - Batch normalization
#    - Weight normalization
#    - Fast dropout (implemented as Gaussian approximation)
#  Tested on CIFAR-100 dataset and simple 3 layer NN:
#      (512-ReLU-512-ReLU-100-Softmax)
#
#  Methods are compared based on:
#    - Performance improvement (over unregularized baseline) based on the amount
#      of data used (accuracy, top-5)
#      [50,125,250,500] images per class
#    - Training speed (time, epochs)
#
#  Experiments are repeated 10 times and their mean and std. deviation are used.
# ------------------------------------------------------------------------------
# --
from __future__ import print_function
# --
import numpy as np
import keras
from keras.datasets import cifar100
import time as t
import cPickle
# --
from models import get_available_models, get_model
# --
SEPARATOR = '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'


def prepare_data(samples_per_class=500):
    """
    Loads the dataset, selects desired number of training samples, and splits
    the test set into test and validation part.

    # Arguments
        samples_per_class: positive int < 500, must divide 500. Determines how
            many samples should be included from each class
    # Returns
        num_classes: int, number of classes (500)
        x_train: np.array of shape (number_of_samples, image_size = 32*32*3)
        y_train: Keras categorical variable of shape (number_of_samples, num_classes)
        x_test, y_test, x_valid, y_valid: like x_train and y_train. Contains half
            of the original test set each
    """
    assert 500 % samples_per_class == 0, \
           'samples_per_class must divide 500 without remainder'

    num_classes = 100

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    # stratified sampling
    train_sorted = np.argsort(y_train, axis=0)
    x_train = x_train[train_sorted]
    y_train = y_train[train_sorted]
    
    test_sorted = np.argsort(y_test, axis = 0)
    x_test = x_test[test_sorted]
    y_test = y_test[test_sorted]

    step = 500/samples_per_class
    x_train, y_train = x_train[::step], y_train[::step]
    x_valid, y_valid = x_test[1::2], y_test[1::2]
    x_test, y_test = x_test[::2], y_test[::2]

    img_size = 32*32*3
    x_train = x_train.reshape(-1, img_size)
    x_test = x_test.reshape(-1, img_size)
    x_valid = x_valid.reshape(-1, img_size)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid = x_valid.astype('float32')
    x_train /= 255
    x_test /= 255
    x_valid /= 255

    assert np.sum(y_train==1)==samples_per_class

    print('>>>> ', x_train.shape[0], 'train samples')
    print('>>>> ', x_test.shape[0], 'test samples')
    print('>>>> ', x_valid.shape[0], 'validation samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    return num_classes, x_train, y_train, x_test, y_test, x_valid, y_valid

def run_experiment(model, data):
    """
    Perform training of the given model on the given data.

    # Arguments
        model: Keras model to train
        data: data structure provided by function prepare_data
    # Returns
        results: dictionary. Contains training history, plus following keys:
            - best_epoch: index of epoch reaching best validation error
            - test_score: test error of the model with the best val. error
            - time: list of timings of each epoch
    """
    num_classes, x_train, y_train, x_test, y_test, x_valid, y_valid = data

    epochs = 1500000/x_train.shape[0]
    batch_size = 128
    best_accuracy = 0.0
    best_epoch = 0
    results = {}
    t_start = t.clock()
    for e in xrange(epochs):
        tic = t.clock()
        hist = model.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=1, verbose=0,
                         validation_data=(x_valid, y_valid))
        toc = t.clock()
        # log time
        hist.history['time'] = [toc-tic]

        # append history to results
        for key in hist.history:
            if key in results:
                results[key].extend(hist.history[key])
            else:
                results[key] = hist.history[key]

        val_acc = hist.history['val_acc'][0]
        if val_acc > best_accuracy:
            best_weights = model.get_weights()
            best_epoch = e + 1 # e is 0-indexed
            best_accuracy = val_acc
    # Evaluate the best model
    model.reset_states()
    model.set_weights(best_weights)
    score = model.evaluate(x_test, y_test, verbose=0)

    # Finalize and return results dict
    results['best_epoch'] = best_epoch
    results['test_score'] = score

    print('>>>>>> Finished. Best model found after ', best_epoch,
          ' epochs. Took ', t.clock()-t_start, 's. Score (loss, top-5, acc) = ', score)
    return results

def run_experiments():
    dataset_sizes = [50,125,250,500]
    repetitions = 10
    experiments = [{'name':model_name,'results':{}} for model_name in get_available_models()]

    for dataset_size in dataset_sizes:
        print(SEPARATOR)
        print('>> Evaluating models on training set with size =', dataset_size)
        print(SEPARATOR)

        print('>>>> Preparing data...')
        data = prepare_data(dataset_size)
        for experiment in experiments:
            print('>>>> Evaluating model', experiment['name'])
            results = []
            for r in xrange(repetitions):
                print('>>>>>> Starting run',r+1,'of', repetitions)
                model = get_model(experiment['name'],data)
                results.append(run_experiment(model, data))
            experiment['results'][dataset_size] = results
    # Now each experiment contains dict of results indexed by dataset_sizes,
    # and each of these contains a list of length repetition with results of
    # each run.
    with open('results.pkl', 'wb') as f:
        cPickle.dump(experiments, f)



# if run from bash
if __name__ == '__main__':
    run_experiments()
