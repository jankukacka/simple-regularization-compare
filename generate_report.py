#!/usr/bin/env python
# ------------------------------------------------------------------------------
#  Comparison of regularization techniqes.
#  Jan Kukacka, 11/2017
#  jan.kukacka@tum.de
# ------------------------------------------------------------------------------
#  Interpretation of the experiment results
# ------------------------------------------------------------------------------

import cPickle
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results(filename):
    assert os.path.isfile(filename), "File " + filename + " does not exist."
    with open(filename) as f:
        results = cPickle.load(f)
    return results

def get_avg_time(histories):
    repetitions = len(histories)
    epochs = len(histories[0]['time'])
    arr = np.empty((repetitions,epochs))
    for i in xrange(repetitions):
        arr[i] = histories[i]['time']
    return np.mean(arr.flatten())

def get_best_test(histories, series_name):
    test_score_index = ['loss', 'acc', 'top_k_categorical_accuracy'].index(series_name)
    repetitions = len(histories)
    arr_ep = np.empty((repetitions,))
    arr_test = np.empty((repetitions,))
    for i in xrange(repetitions):
        arr_ep[i] = histories[i]['best_epoch']
        arr_test[i] = histories[i]['test_score'][test_score_index]
    ep_mean = np.mean(arr_ep)
    ep_std = np.std(arr_ep)
    score_mean = np.mean(arr_test)
    score_std = np.std(arr_test)
    return score_mean, score_std, ep_mean, ep_std

def plot_series(res, axes=None, model_index=0, dataset_size=500,
                series_name='loss', timed=False, show_best_test=False,
                line_params={}, dev_params={}):
    histories = res[model_index]['results'][dataset_size]
    repetitions = len(histories)
    epochs = len(histories[0][series_name])
    arr = np.empty((repetitions, epochs))
    for i in xrange(repetitions):
        arr[i] = histories[i][series_name]
    means = np.mean(arr, axis=0)
    devs = np.std(arr, axis=0)

    timing = xrange(epochs)
    if timed:
        avg_time = get_avg_time(histories)
        timing = np.array(timing)*avg_time

    if axes is None:
        axes = plt.gca()
    if 'label' not in line_params:
        line_params['label'] = res[model_index]['name'],
    axes.plot(timing, means, **line_params)
    axes.fill_between(timing, means+devs, means-devs, **dev_params)
    axes.legend(loc='upper left', bbox_to_anchor=(1,1))
    if show_best_test:
        best_test_mean, best_test_std, best_ep_mean, best_ep_std = get_best_test(histories, series_name)
        timing = 1
        if timed:
            timing = get_avg_time(histories)
        axes.errorbar(best_ep_mean*timing, best_test_mean,
                      xerr=best_ep_std*timing, yerr=best_test_std,
                      marker='x', markersize=10, c=line_params['color'])

def plot_comparison(res, dataset_size, title, series_name, xlabel='epochs',
                    timed=False, save_png=True, save_eps=True):
    plt.figure()
    plt.title(title)
    plt.gca().set_ylabel(series_name)
    plt.gca().set_xlabel(xlabel)
    colors = 'rgbc'
    for i in xrange(len(res)):
        color = colors[i]
        line_params = {'label': res[i]['name'] + ' (training)', 'color': color}
        dev_params = {'linewidth': 0, 'alpha': 0.3, 'facecolor': color}
        plot_series(res, model_index=i, series_name=series_name,
                    dev_params=dev_params, line_params=line_params,
                    dataset_size=dataset_size, timed=timed, show_best_test=True)

        line_params = {'label': res[i]['name'] + ' (validation)', 'color': color, 'linestyle':'--'}
        plot_series(res, model_index=i, series_name='val_'+series_name,
                    dev_params=dev_params, line_params=line_params,
                    dataset_size=dataset_size, timed=timed)
    if save_png:
        plt.savefig('report/plot_'+series_name+'_{}.png'.format(dataset_size),
                    bbox_inches='tight')
    if save_eps:
        plt.savefig('report/plot_'+series_name+'_{}.eps'.format(dataset_size),
                    bbox_inches='tight')
    plt.close()


def generate_report(results_filename='report/results.pkl', generate_png=True,
                    generate_eps=True):
    res = load_results(results_filename)
    # Folder for saving images
    if not os.path.isdir('report'):
        os.mkdir('report')

    # normal charts
    for key in sorted(res[0]['results']):
        plot_comparison(res, key, 'Loss, {} samples'.format(key), 'loss',
                        save_png=generate_png, save_eps=generate_eps)
        plot_comparison(res, key, 'Accuracy, {} samples'.format(key), 'acc',
                        save_png=generate_png, save_eps=generate_eps)
        plot_comparison(res, key, 'Top-5, {} samples'.format(key),
                        'top_k_categorical_accuracy',save_png=generate_png,
                        save_eps=generate_eps)

    # timed charts
    for key in sorted(res[0]['results']):
        plot_comparison(res, key, 'Loss, {} samples'.format(key), 'loss',
                        timed=True, xlabel='time (s)', save_png=generate_png,
                        save_eps=generate_eps)
        plot_comparison(res, key, 'Accuracy, {} samples'.format(key), 'acc',
                        timed=True, xlabel='time (s)', save_png=generate_png,
                        save_eps=generate_eps)
        plot_comparison(res, key, 'Top-5, {} samples'.format(key),
                        'top_k_categorical_accuracy', timed=True,
                        xlabel='time (s)',
                        save_png=generate_png,
                        save_eps=generate_eps)

if __name__ == '__main__':
    generate_report()


#----------------------------------------------------------------
# Helping function for printing numeric data
def print_best_epoch(res, model_index, dataset_size):
    # Convergence speed metrics
    for dataset_size in sorted(res[0]['results']):
        print 'Dataset size:', dataset_size
        for model_index in xrange(len(res)):
            histories = res[model_index]['results'][dataset_size]
            repetitions = len(histories)
            arr = np.empty((repetitions,))
            for i in xrange(repetitions):
                arr[i] = histories[i]['best_epoch']
            means = np.mean(arr)
            devs = np.std(arr)

            print res[model_index]['name'], 'converges in {:.2f}+-{:.4f}'.format(means, devs)

def print_best_scores(res, model_index, dataset_size):
    for dataset_size in sorted(res[0]['results']):
        print 'Dataset size:', dataset_size
        for model_index in xrange(len(res)):
            histories = res[model_index]['results'][dataset_size]
            repetitions = len(histories)
            arr = np.empty((repetitions,3))
            for i in xrange(repetitions):
                arr[i] = histories[i]['test_score']
            means = np.mean(arr, axis=0)
            devs = np.std(arr, axis=0)

            print '. ', res[model_index]['name'], 'has loss {:.2f} +- {:.4f}'.format(means[0], devs[0])
            print '. ', res[model_index]['name'], 'has accuracy {:.1f}% +- {:.2f}%'.format(means[1]*100, devs[1]*100)
            print '. ', res[model_index]['name'], 'has top-5 {:.1f}% +- {:.2f}%'.format(means[2]*100, devs[2]*100)
            print '..'

def print_timings(res, model_index, dataset_size):
    for dataset_size in sorted(res[0]['results']):
        print 'Dataset size:', dataset_size
        for model_index in xrange(len(res)):
            histories = res[model_index]['results'][dataset_size]
            repetitions = len(histories)
            epochs = len(histories[0]['time'])
            arr = np.empty((repetitions,epochs))
            for i in xrange(repetitions):
                arr[i] = histories[i]['time']
            means = np.mean(arr.flatten())
            devs = np.std(arr.flatten())

            print '. ', res[model_index]['name'], 'needs {:.2f}s +- {:.4f}s'.format(means, devs), 'per epoch'
