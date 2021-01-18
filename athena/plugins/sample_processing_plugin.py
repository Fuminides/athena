# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:16:19 2020

@author: Javier Fumanal Idocin (Fuminides)
"""
import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

import Fancy_aggregations as fz
import bci_architectures as athena

# =============================================================================
#   	CLASSIC OVERSAMPLING METHODS - The names are self-descriptive
# =============================================================================
def generic_sample(X, y, sample_alg):
    X_res = []
    y_res = []
    for X_wave in X:
        X_resampled, y_resampled = sample_alg().fit_resample(X_wave, y)
        X_res.append(X_resampled)
        y_res.append(y_resampled)

    return X_res, y_res

def smote_samples(X, y):
    '''
    X is (features, samples)
    y is (samples,)
    '''
    return generic_sample(X, y, SMOTE)

def adasyn_samples(X, y):
    '''
    X is (features, samples)
    y is (samples,)
    '''
    return generic_sample(X, y, ADASYN)


def borderline_smote_samples(X, y):
    '''
    X is (features, samples)
    y is (samples,)
    '''
    return generic_sample(X, y, BorderlineSMOTE)

def smote_tomek_samples(X, y):
    '''
    X is (features, samples)
    y is (samples,)
    '''
    return generic_sample(X, y, SMOTETomek)

def smoteen_samples(X, y):
    '''
    X is (features, samples)
    y is (samples,)
    '''
    return generic_sample(X, y, SMOTEENN)

data_preprocessing_dict = {
    'smote_': smote_samples, 'adasyn' : adasyn_samples,
    'border': borderline_smote_samples,
    'tomek': smote_tomek_samples, 'smoteen': smoteen_samples}

# =============================================================================
# SIGNAL PROCESSING OVERSAMPLE
# =============================================================================
def signal_boltzmann_reduction(X, y, layers_size=(30, 20, 10), sample_rate=5):
    '''
    Boltzmann machine of type feature transformer.
    It trains supervisedly and returns the transformed features.


    :param X: (bands, time/features, samples)
    :param sample_rate: number of times to multiply the existing samples.
    :returns : (bands, samples, features)
    '''
    from boltzmann_machine_deep import DBN

    bands, time, samples = X.shape
    machines = []

    X_transformed = np.zeros((bands, layers_size[-1], samples))

    for freq in np.arange(bands):
        X_bands = X[freq, :, :]
        X_reshaped = X_bands.T
        machine = DBN(n_neurons=layers_size, supervised=False)
        machine.fit(X_reshaped>0, y)

        machines.append(machine)
        X_transformed[freq, :, :] = machine.transform(X_reshaped>0).T

    return X_transformed, y

def random_noise(X, y, sample_rate=5, random_distribution='normal'):
    '''
    Generates samples by adding random noise to the differentiated signal.

    :param sample_rate: number of times to multiply the existing samples.
    :param random_distribution: if 'normal': gaussian noise. If uniform then
    the noise is drawn from a random uniform distribution

    :returns : (bands, samples, features)
    '''
    freqs, wave_size, samples = X.shape
    res = np.zeros((freqs, wave_size, samples * sample_rate))
    y_new = np.zeros((samples * sample_rate))

    if random_distribution == 'normal':
        gaussian_noise = np.random.normal(0 ,0.05, (freqs, wave_size, samples * (sample_rate)))
    elif random_distribution == 'uniform':
        gaussian_noise = np.random.uniform(0 ,0.05, (freqs, wave_size, samples * (sample_rate)))


    for i in range(sample_rate):
        target = np.arange(i*samples, (i+1)*samples)
        if i == 0:
            new = X
        else:
            new = X + gaussian_noise[:, :, target]

        res[:, :, target] = new
        y_new[target] = y

    return res, y_new

def boltzmann_augmentation(X, y, sample_rate=5, kinetic=False):
    '''
    Generates samples by training a kinetic ising model.

    :param sample_rate: number of times to multiply the existing samples.
    :param kinetic: if true then it is a konetic ising, if false it a static ising.
    (Note: do not use the static unless you really know what you are doing)

    :returns : (bands, samples, features)
    '''
    import wormbrain
    freqs, wave_size, samples = X.shape
    avg_freq = np.mean(X, axis=2)
    std_freq = np.std(X, axis=2)

    def bool_to_signal(bool_signal):
        '''
        Sicne the boltzmann machine returns booleans, we need to convert that
        in a real "smooth" signal. To do so:
            -We interpret the booleans as growth signal (positive or negative)
            -Then we use a random number [0,1] to multiply each number. Note: posibility to learn the real pdf.
            -Voila.
        '''
        w, f = bool_signal.shape

        signal_growths = bool_signal * 2 - 1
        for time_w in range(w):
            for freq_x in range(f):
                random_correction = np.random.normal(avg_freq[freq_x, time_w], std_freq[freq_x, time_w], 1)
                signal_growths[time_w, freq_x] = signal_growths[time_w, freq_x] * random_correction

        return signal_growths

    def train_isings_class(x_class):
        '''
        Trains an Ising model for each frequency studied in the training data.

        :param x_class: training data (features, time, samples)
        :return : augmented data.
        '''
        freqs0, wave_size0, samples0 = x_class.shape
        reshaped = np.zeros((wave_size0 * samples0, freqs0))

        for fr in range(freqs0):
            reshaped[:, fr] = x_class[fr, :, :].reshape((wave_size0 * samples0))

        isings, fits = wormbrain.analyze_model.train_ising([reshaped], kinetic=kinetic)
        model = isings[0]

        m_list = [bool_to_signal(model.generate_sample(wave_size0, booleans=True)) for times in range(samples0*(sample_rate-1))]
        res = np.zeros((freqs0, wave_size0, len(m_list)))
        for mx, m in enumerate(m_list):
            res[:, :, mx] = np.swapaxes(m,0,1)

        return res


    n_clases = len(np.unique(y))
    res = np.zeros((freqs, wave_size, samples * sample_rate))
    y_new = np.zeros((samples * sample_rate))
    start = 0
    for ix, clase in enumerate(range(n_clases)):
        index_class = y == clase
        x = X[:, :, index_class]

        new_muestras = train_isings_class(x)
        freqs0, wave_size0, samples0 = new_muestras.shape
        target = np.arange(start, start + samples0)
        start += samples0

        res[:, :, target] = new_muestras
        y_new[target] = clase

    res[:, :, start:] = X
    y_new[start:] = y

    return res, y_new

signal_preprocessing_dict = {
    'random' : random_noise,
    'uniform' : lambda X, y, sample_rate=5: random_noise(X, y, sample_rate=sample_rate, random_distribution='uniform'),
    'transform' : signal_boltzmann_reduction,
    'kinetic': lambda X, y, sample_rate=5: boltzmann_augmentation(X, y, sample_rate, True)}

# =============================================================================
# SAMPLE PROCESSING ARCHITECTURES
# =============================================================================
class bci_achitecture_process(athena.bci_achitecture):
    '''
    Additional architectures for the sampling processing plugin. They implement
    the basic traditional and enhanced-traditional frameworks as described in the
    Transactions on Cybernetics paper. But they include support for the augmentation
    techniques.
    '''
    def trad_process_architecture(self, X, y, verbose=False, agregacion=None, preprocess_function_signal=None, preprocess_function_classifier=None):
            '''
            Traditional BCI framework with signal-based and stats-based sample augmentation techniques.

            :param X: train (bands, time, samples)
            :param agregacion: tuple of (aggregations, classifier type). If not given, will be learnt.
            :param preprocess_function_signal: signal-based augmentation function. None by deafult.
            :param preprocess_function_classifier: stat-based augmentation function. None by deafult.
            :param y: labels (samples,)
            '''
            def _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes):
                csp_x = athena._csp_forward(Xf, csp_models)
                logits = athena._fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)

                return ag1(logits, axis=0, keepdims=False)
            try:
                if len(agregacion) == 2:
                    agregacion, clasificador = agregacion
            except TypeError:
                clasificador = None
            self._num_classes = len(np.unique(y))

            if not (preprocess_function_signal is None):
                X, y = preprocess_function_signal(X, y)

            csp_models, csp_x = athena._csp_train_layer(X, y)

            csp_og = csp_x
            y_og = y

            if not (preprocess_function_classifier is None):
                csp_x, y = preprocess_function_classifier(csp_x, y)

            classifier = athena._trad_classifier_train(csp_x, y, designed_classifier=clasificador)

            logits = athena._fitted_trad_classifier_forward(csp_og, classifier, self._num_classes)

            if agregacion is None:
                ag1 = athena._decision_making_learn(logits, y_og)
            else:
                try:
                    ag1 = fz.binary_parser.parse(agregacion)
                except AttributeError:
                    ag1 = agregacion

            self._components = [csp_models, classifier, ag1, self._num_classes]
            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifier, ag1, self._num_classes)

    def emf_process_sample_architecture(self, X, y, verbose=False, agregacion=None, preprocess_function_signal=None, preprocess_function_classifier=None):
        '''
            Traditional BCI framework with signal-based and stats-based sample augmentation techniques.

            :param X: train (bands, time, samples)
            :param agregacion: tuple of (aggregations, classifier type). If not given, will be learnt.
            :param preprocess_function_signal: signal-based augmentation function. None by deafult.
            :param preprocess_function_classifier: stat-based augmentation function. None by deafult.
            :param y: labels (samples,)
            '''
        def _emf_predict_proba(X, csp_models, classifiers, ag1, ag2, num_classes):
            csp_x = athena._csp_forward(X, csp_models)
            logits = athena._fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

            return athena._multimodal_fusion(logits, ag1, ag2, num_classes)

        self._num_classes = len(np.unique(y))

        if not (preprocess_function_signal is None):
            X, y = preprocess_function_signal(X, y, sample_rate=2)

        csp_models, csp_x = athena._csp_train_layer(X, y)

        csp_og = csp_x
        y_og = y

        if not (preprocess_function_classifier is None):
            csp_x, y = preprocess_function_classifier(csp_x, y)

        classifiers = athena._classifier_std_block_train(csp_x, y)

        logits = athena._fitted_classifiers_forward(csp_og, classifiers, self._num_classes)

        ag1, ag2 = athena._multimodal_fusion_learn(logits, y_og)

        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, ag1, ag2, self._num_classes)