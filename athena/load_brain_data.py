# -*- coding: utf-8 -*-
"""
Functions to load experimentation data for the BCI experiments.

Created on Thu Mar 28 15:26:53 2019

@author: javi-
"""
import os
import pickle

import scipy.io
from mne.decoding import CSP
import numpy as np
import pandas as pd

##############################
#   WAVE DEFINITIONS
##############################
delta = np.arange(0, 3, dtype=np.int8)
theta = np.arange(3, 7, dtype=np.int8)
alpha = np.arange(7, 13, dtype=np.int8)
beta = np.arange(13, 30, dtype=np.int8)
all_waves = np.arange(0, 30, dtype=np.int8)
sensory_motor = np.arange(12, 15, dtype=np.int8)

#All but 3 and 5. For the UTS data look for the IEEE Cybernetics and Com. Intelligence papers.
good_subjects_uts_data_uts_data = ['1', '2', '4', '6', '7',
                 '8', '9', '10', '11', '12', '13',
                 '14', '15', '16', '17', '18', '19', '20']

#Subjects for the 2a BCI IV dataset
d2_a = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
wavelets = (delta, theta, alpha, beta)

tasks = {'L/R':0, 'L/F':1, 'L/T':2, 'R/F':3, 'R/T':4, 'F/T':5, 'All': 0}
##############################

def load_all_bci2a(basal_adjust=True, signal_filter=False, normalize=False, derivate=2, transition_time=0):
    '''
    Load all good quality subjects from the experiment. (See load_subject() for more info)

    :basal_adjust: if true, will substract the mean of the basal activity (first 500 observations) to each subject.
    :signal_filter: if true, will perform a LOWESS filter to the signal.
    :normalize: normalize the subject by zscore.

    :return: A tuple (X, y), where X corresponds to each subject data and y to its labels.
    '''
    global d2_a

    y_res = np.empty([0], np.int)
    first = True

    for ix, element in enumerate(d2_a):
        sX = load_competition_subject(element)
        sX, y = join_labels_BCI(sX)

        if basal_adjust:
            sX = sX[:, 250:1000, :]

        if normalize:
            mean_x = np.mean(np.mean(sX, axis=1, keepdims=True), axis=1, keepdims=True)
            std_x = np.std(np.std(sX, axis=1, keepdims=True), axis=1, keepdims=True)
            sX = (sX - mean_x) / std_x

        if signal_filter:
            from scipy.signal import savgol_filter
            sX = sX - savgol_filter(sX, 31, 2, axis=1)

        for n_diff in range(derivate):
            sX = np.diff(sX, n=1, axis=1, prepend=0)

        if transition_time > 0:
            sX = sX[:, [int(x) for x in range(sX.shape[1]) if int(x % transition_time) == 0], :]

        if first:
            final_res = np.empty([sX.shape[0], sX.shape[1], 0], np.float, )
            final_res = np.append(final_res, sX, axis=2)
            first = False
        else:
            final_res = np.append(final_res, sX, axis=2)

        y_res = np.append(y_res, y)

    return final_res, y_res

def load_all_bci2b(basal_adjust=True, signal_filter=False, normalize=False, derivate=1, transition_time=0):
    '''
    Load all good quality subjects from the experiment. (See load_subject() for more info)

    :basal_adjust: if true, will substract the mean of the basal activity (first 500 observations) to each subject.
    :signal_filter: if true, will perform a LOWESS filter to the signal.
    :normalize: normalize the subject by zscore.

    :return: A tuple (X, y), where X corresponds to each subject data and y to its labels.
    '''
    global d2_a

    y_res = np.empty([0], np.int)
    first = True

    for ix, element in enumerate(d2_a):
        for test in range(1, 4):
            sX = load_competition_subject(element, version='b', test=str(test))
            sX, y = join_left_right_BCI(sX)

            if basal_adjust:
                sX = sX[:, 200:, :]

            if normalize:
                mean_x = np.mean(np.mean(sX, axis=2, keepdims=True), axis=1, keepdims=True)
                std_x = np.std(np.std(sX, axis=2, keepdims=True), axis=1, keepdims=True)
                sX = (sX - mean_x) / std_x

            if signal_filter:
                from scipy.signal import savgol_filter
                sX = sX - savgol_filter(sX, 31, 2, axis=1)

            for n_diff in range(derivate):
                sX = np.diff(sX, n=1, axis=1, prepend=0)

            if transition_time > 0:
                sX = sX[:, [int(x) for x in range(sX.shape[1]) if int(x % transition_time) == 0], :]

            if first:
                final_res = np.empty([sX.shape[0], sX.shape[1], 0], np.float, )
                final_res = np.append(final_res, sX, axis=2)
                first = False
            else:
                final_res = np.append(final_res, sX, axis=2)

            y_res = np.append(y_res, y)

    return final_res, y_res

def load_datasets_bci2a(basal_adjust=False, signal_filter=False, normalize=True, derivate=2,
                        transition_time=0, tongue_feet=False, cache_mode=True):
    '''
    Load all good quality subjects from the experiment. (See load_subject() for more info)

    :basal_adjust: if true, will substract the mean of the basal activity (first 500 observations) to each subject.
    :signal_filter: if true, will perform a LOWESS filter to the signal.
    :normalize: normalize the subject by zscore.

    :return: A tuple (X, y), where X corresponds to each subject data and y to its labels.
    '''
    global d2_a
    cache_file = 'bci_iv_logits/bci_iv_datasets.pckl'
    cache_file_tongue_feet = 'bci_iv_logits/bci_iv_datasets_four_classes.pckl'
    if tongue_feet:
            cache_file = cache_file_tongue_feet

    if cache_mode and os.path.isfile(cache_file):
        with open(cache_file, 'rb') as fb:
            datasets = pickle.load(fb)

    else:
        datasets = []

        for ix, element in enumerate(d2_a):
            y_res = np.empty([0], np.int)
            first = True
            for test in range(1, 4):
                sX = load_competition_subject(element, version='a', test=str(test), tongue_feet=tongue_feet)
                sX, y = join_labels_BCI(sX, tongue_feet=tongue_feet)
                if basal_adjust:
                    sX = sX[:, 250:1000, :]

                if normalize:
                    mean_x = np.mean(sX, axis=1, keepdims=True)
                    std_x = np.std(sX, axis=1, keepdims=True)
                    sX = (sX - mean_x) / std_x

                if signal_filter:
                    from scipy.signal import savgol_filter
                    sX = sX - savgol_filter(sX, 31, 2, axis=1)

                for n_diff in range(derivate):
                    sX = np.diff(sX, n=1, axis=1, prepend=0)

                if transition_time > 0:
                    sX = sX[:, [int(x) for x in range(sX.shape[1]) if int(x % transition_time) == 0], :]

                if first:
                    final_res = np.empty([sX.shape[0], sX.shape[1], 0], np.float, )
                    final_res = np.append(final_res, sX, axis=2)
                    first = False
                else:
                    final_res = np.append(final_res, sX, axis=2)

                y_res = np.append(y_res, y)

            datasets.append((final_res, y_res))

        if cache_mode:
            with open(cache_file, 'wb') as fb:
                pickle.dump(datasets, fb)


    return datasets

def load_competition_subject(name = '1', version='a', test='1', tongue_feet=False, dataset_path='BCI Competition IV dataset 2a/A0'):
    if version == 'a':
        try:
            a = scipy.io.loadmat(dataset_path + name + 'TClass1stft.mat')['stft_data'] #They must have the FFT already done!
            b = scipy.io.loadmat(dataset_path + name + 'TClass2stft.mat')['stft_data']
        except FileNotFoundError:
            a = scipy.io.loadmat(dataset_path + name + 'TClass1_stft.mat')['stft_data']
            b = scipy.io.loadmat(dataset_path + name + 'TClass2_stft.mat')['stft_data']

        if tongue_feet:
            c = scipy.io.loadmat(dataset_path + name + 'TClass3stft.mat')['stft_data']
            d = scipy.io.loadmat(dataset_path + name + 'TClass4stft.mat')['stft_data']

            return a, b, c, d
        else:
            return a,b
    else:
        a = scipy.io.loadmat('BCI Competition IV dataset 2b/B0' + name + '0'+ test +'TClass1stft.mat')['stft_data']
        b = scipy.io.loadmat('BCI Competition IV dataset 2b/B0' + name + '0'+ test +'TClass2stft.mat')['stft_data']

        return a,b


def fourier_transform(X):
    '''
    Comptues the fast fourier transform. Technically, it reproduces the spectrogram
    matlab function behaviour.
    Note: I strongly recommend to use the matlab function and save the results
    directly in the files. This function has not been properly tested and it's behavior
    might not be exactly as intended.
    '''
    from matplotlib import mlab

    fs = 250
    winlen = 50
    overlap = 40

    s, _, _ = np.real(mlab.specgram(X, Fs = fs, NFFT = winlen, noverlap = overlap, mode='power'))

    return np.pad(s, (len(X)- s.shape[1],0), mode='edge')


def extract_test(experimentation, n):
    '''
    Returns a the n-ths experimentations from a single subject.

    :param experimentation: a tuple containing right hand and left hand experiments.
    :param n: n-th experiment
    :return: (a tuple (Experiment right (30x3000), Experiment left (30x3000)))
    '''
    left, right = experimentation
    return left[:,:,n], right[:,:,n]

def calc_CSP(experiment, bands, agg_waves=None, n_filters=25):
    '''
    Calculate the CSP for the set of experimentations for a single subject.

    :param experiment: a tuple containing right hand and left hand experiments.
    :return: the calculated csp for each subject.
    '''
    from contextlib import contextmanager,redirect_stderr,redirect_stdout
    from os import devnull

    @contextmanager
    def suppress_stdout_stderr():
        """A context manager that redirects stdout and stderr to devnull"""
        with open(devnull, 'w') as fnull:
            with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                yield (err, out)
    with suppress_stdout_stderr():
        csp = CSP(n_filters)
        if len(experiment[1].shape) == 1:
            complete, y = experiment
        else:
            complete, y = join_left_right_BCI(experiment)

        if agg_waves is not None:
            complete = np.transpose(np.mean(complete[bands, :, :],axis=0, keepdims=True), (2, 0, 1))
        else:
            complete = np.transpose(complete[bands, :, :], (2, 0, 1))
        try:
            csp = csp.fit(complete, y)
        except IndexError:
            csp = csp.fit(complete, y)

        return csp.transform(complete), y, csp

def calc_std_CSP(experiment, y=None, n_filters=25):
    '''
    Calculates CSP from a experiment data using the standard wavelets.

    :experiment: check the result of load_all()
    :y: labels. If not given, calc_CSP() will infer them (requires experiments correctly sorted).
    :return: 5 csp matrix (FiltersxSamples), the labels and a tuple with the csp models.
    '''
    if y is None:
        x = experiment
    else:
        x = (experiment, y)

    csp1, y, csp1m = calc_CSP(x, delta, n_filters=n_filters)
    csp2, _, csp2m = calc_CSP(x, theta, n_filters=n_filters)
    csp3, _, csp3m = calc_CSP(x, alpha, n_filters=n_filters)
    csp4, _, csp4m = calc_CSP(x, beta, n_filters=n_filters)
    csp5, _, csp5m = calc_CSP(x, all_waves, n_filters=n_filters)
    csp_senmot, _, csp_senmot_m = calc_CSP(x, sensory_motor, n_filters=n_filters)

    return csp1, csp2, csp3, csp4, csp5, csp_senmot, y, (csp1m, csp2m, csp3m, csp4m, csp5m, csp_senmot_m)

def apply_CSP(X, models, bands=(delta, theta, alpha, beta, all_waves, sensory_motor)):
    '''

    :param X: BCI datain the shape (frequency bands, samples, classes).
    :param models: trained csp models.
    :param bands: a list of the different frequency wave bands to compute the CSP.
    :return: The CSP tranformation for each wave bands.
    '''
    X = np.transpose(X, (2, 0, 1))

    return [models[x].transform(X[:, bands[x], :]) for x in range(len(models))]

def join_left_right_BCI(experiment):
    '''
    Joins the right and left or right/left experiments
    into one single numpy array.

    :param experiment: tuple containing the arrays for each experiment.
    :return: a tuple with the features and label for each sample.
    '''
    return join_left_right_BCI(experiment, False)

def join_labels_BCI(experiment, tongue_feet=False):
    '''
    Joins the right and left or right/left/feet/tongue experiments
    into one single numpy array.

    :param experiment: tuple containing the arrays for each experiment.
    :return: a tuple with the features and label for each sample.
    '''
    if tongue_feet:
        left, right, foot, tongue = experiment
        complete = np.append(left, right, axis=2)
        complete = np.append(complete, foot, axis=2)
        complete = np.append(complete, tongue, axis=2)
    else:
        left, right = experiment
        complete = np.append(left, right, axis=2)


    y1 = np.zeros(left.shape[2])
    y2 = np.ones(right.shape[2])

    y = np.append(y1, y2)
    if tongue_feet:
        y3 = np.ones(foot.shape[2]) * 2
        y4 = np.ones(tongue.shape[2]) * 3
        y = np.append(y, y3)
        y = np.append(y, y4)

    return complete, y


def obtain_standard_waves(csp_data):
    '''
    Returns the different wavelets for a set of subjects.

    :param csp_data: csp data in the form 30xS (xclases optional) (S i s the number of subjects)
    :return: a tuple with the different wave sets.
    '''
    global delta, theta, alpha, beta

    if len(csp_data.shape) == 2:
        delta_wavelet = csp_data[delta, :]
        theta_wavelet = csp_data[theta, :]
        alpha_wavelet = csp_data[alpha, :]
        beta_wavelet = csp_data[beta, :]
        all_wavelet = csp_data[all_waves, :]
    elif len(csp_data.shape) == 3:
        delta_wavelet = csp_data[delta, :, :]
        theta_wavelet = csp_data[theta, :, :]
        alpha_wavelet = csp_data[alpha, :, :]
        beta_wavelet = csp_data[beta, :, :]
        all_wavelet = csp_data[all_waves, :, :]

    return delta_wavelet, theta_wavelet, alpha_wavelet, beta_wavelet, all_wavelet

def temporal_series_analyze(subject, onda=[0], verbose=True, diff=True):
    '''
    Returns for a subject the decomposition of it's wave in time domain into
    tendency, stationary and random components.

    :param subject: data to analyze
    :param onda: wave freq to decompose
    :param verbose: if True, will plot the results.
    :param diff: if True, will differentiate the signal.
    :return : the decomposed signal for the right and left trials in a tuple of 2.
    '''
    from statsmodels.tsa.seasonal import seasonal_decompose

    if not diff:
        test_izq = np.mean(subject[0][onda,:,:], axis=2)
        test_dr = np.mean(subject[1][onda,:,:], axis=2)
    else:
        test_izq = np.mean(np.diff(subject[0][onda,:,:],axis=1), axis=2)
        test_dr = np.mean(np.diff(subject[1][onda,:,:],axis=1), axis=2)

    test_izq = np.mean(test_izq,axis=0)
    test_dr = np.mean(test_dr,axis=0)

    series_izq = pd.Series(test_izq)
    series_dr = pd.Series(test_dr)

    if diff:
        series_izq.index = pd.DatetimeIndex(freq='w', periods=2999, start=0)
        series_dr.index = pd.DatetimeIndex(freq='w', periods=2999, start=0)
    else:
        series_izq.index = pd.DatetimeIndex(freq='w', periods=3000, start=0)
        series_dr.index = pd.DatetimeIndex(freq='w', periods=3000, start=0)

    decom1 = seasonal_decompose(series_izq,model='additive', extrapolate_trend='freq')
    decom2 = seasonal_decompose(series_dr,model='additive', extrapolate_trend='freq')

    if verbose:
        from matplotlib import pyplot as plt
        #decom1.plot()
        if not diff:
            x_axis = np.linspace(0, 14, 3000)
        else:
            x_axis = np.linspace(0, 14, 2999)

        plt.figure()
        plt.plot(x_axis, decom1.observed.values, label='Left trial')
        plt.plot(x_axis, decom2.observed.values, label='Right trial', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('V')
        plt.legend(loc="upper left")
        if not diff:
            plt.savefig('observed.pdf'); plt.show()
        else:
            plt.savefig('observed_diff.pdf'); plt.show()

        plt.figure()
        plt.plot(x_axis, decom1.trend.values, label='Left trial')
        plt.plot(x_axis, decom2.trend.values, label='Right trial', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('V')
        plt.legend(loc="upper left")
        if not diff:
            plt.savefig('trend.pdf'); plt.show()
        else:
            plt.savefig('trend_diff.pdf'); plt.show()

        plt.figure()
        plt.plot(x_axis, decom1.seasonal.values, label='Left trial')
        plt.plot(x_axis, decom2.seasonal.values, label='Right trial', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('V')
        plt.legend(loc="upper left")
        if not diff:
            plt.savefig('station.pdf'); plt.show()
        else:
            plt.savefig('station_diff.pdf'); plt.show()

        plt.figure()
        plt.plot(x_axis, decom1.resid.values, label='Left trial')
        plt.plot(x_axis, decom2.resid.values, label='Right trial', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('V')
        plt.legend(loc="upper left")
        if not diff:
            plt.savefig('random.pdf'); plt.show()
        else:
            plt.savefig('random_diff.pdf'); plt.show()
    else:
        return (decom1, decom2)


