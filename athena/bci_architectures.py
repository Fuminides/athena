# -*- coding: utf-8 -*-

"""
Functions to load basic blocks and architectures (frameworks) for BCI experiments.
It is based on the Enhanced Multimodal Fusion Framework works, and their subsqeunt papers.

Created on Thu Mar 28 15:26:53 2019

arXiv:
    Fumanal-Idocin, J., Wang, Y., Lin, C., Fern'andez, J., Sanz, J.A., & Bustince, H. (2021).
    Motor-Imagery-Based Brain Computer Interface using Signal Derivation and Aggregation Functions.

@author: Javier Fumanal Idocin (Fuminides)
"""
import numpy as np

from . import load_brain_data as lb
import Fancy_aggregations.binary_parser as bp

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

classifier_types = [SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier, GaussianProcessClassifier]

agg_functions_names = ['sugeno', 'shamacher', 'choquet', 'cfminmin', 'cf12', 'cf', 'owa1', 'owa2', 'owa3', 'geomean', 'sinoverlap',
                'hamacher', 'luka', 'probabilistic_sum', 'bounded_sum', 'einstein_sum']

mff_aggregations = ['choquet', 'sugeno', 'shamacher', 'cfminmin']
classical_aggregations = ['mean', 'min', 'max', 'median']


def _fast_montecarlo_optimization(function_alpha, x0=[0.5], minimizer_kwargs=None, niter=200, smart=True):
    '''
    Just randomly samples the function. More functionality might come in the future if necessary.
    Called from weight_numerical_optimization.
    '''
    class dummy_plug:
        def _init(self, x=None):
            self.x = x

    iter_actual = 0

    #epsilon = 0.05
    eval_per_iter = 5
    best_fitness = 1
    resultado = dummy_plug()
    if hasattr(x0, '__len__'):
        size_2 = len(x0)
    else:
        size_2 = 1
    while(iter_actual < niter):
        subjects = np.random.random_sample((eval_per_iter, size_2))
        fitness = [function_alpha(x) for x in subjects]
        ordered = np.sort(fitness)
        arg_ordered = np.argsort(fitness)
        iter_actual += 1

        if ordered[1] < best_fitness:
            best_fitness = ordered[1]
            resultado.x = subjects[arg_ordered[1], :]
            resultado.fun = best_fitness
            if best_fitness == 0.0:
                return resultado

    return resultado

def weight_numerical_optimization(func, x0=0.5, bounds=[0, 1], niter=100, mode='montecarlo', verbose=False):
    '''
    Optimize the func using numerical optimization techniques: trivial values, bashinhopping and
    naive montecarlo.

    :param func: function to optimize.
    :param x0: initial subject(s), must be valid input to func.
    :param bounds: constrains for each value in the subjects.
    :param niter: number of iterations/sampling in the methods.
    :param mode: algorithm to use: {trivial, montecarlo, basinhopping}. Trivial just looks for some values in [0,1] range.
    :param verbose: if True, shows a report of the solution found on screen.
    '''
    from scipy.optimize import  basinhopping
    import time

    try:
        len_args = len(x0)
    except TypeError:
        len_args = 1

    bounds = {'bounds': [bounds] * len_args}
    start = time.time()
    if mode == 'basinhopping':
        res = basinhopping(func, x0=x0, niter=niter, minimizer_kwargs=bounds, seed=4)

    elif mode == 'trivial':
        class dummy_plug:
            def _init(self, x=None):
                self.x = x

        res = dummy_plug()
        if len_args > 1:
            candidates = [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]
            res.x = [0.5, 0.5] #Default result
        else:
            candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
            res.x = [0.5] #Default result

        best = 0.0
        for candidate in candidates:
            acc = func(candidate)

            if acc > best:
                res.x = candidate
                res.fun = acc
                best = acc

    elif mode == 'montecarlo':
        res = _fast_montecarlo_optimization(func, x0=x0, niter=niter)
    end = time.time()

    if verbose:
        print('Result: ' + str(res.x), 'Time taken: ' + str(end - start), 'Error: ' + str(res.fun) + '(Acc: ' + str(1 - res.fun) + ')')

    return res.x
# =============================================================================
#   BASIC COMPONENTS
# =============================================================================
def decision_making_learn(X, y):
    '''
    Learns the best aggregation function for a set of input data and their labels.
    Uses the accuracy as evaluation criteria.

    :param X: intput training data of shape (classifiers, samples, clases)
    :param y: labels array for each sample.
    :return : an aggregation function.
    '''
    def compute_accuracy(yhat, y):
        return np.mean(yhat == y)
    maximo = 0
    mejor_aggregacion = None

    for agg_f0 in agg_functions_names:
        agg_f = bp.parse(agg_f0)
        acc = compute_accuracy(np.argmax(agg_f(X, axis=0, keepdims=False),axis=1), y)

        if acc > maximo:
            maximo = acc
            mejor_aggregacion = agg_f

    return mejor_aggregacion



def classifier_std_block_train(csp_x, labels):
    '''
    Trains a list of list classifiers from the extracted features obtained from
    transforming the original FFT/EEG waves. So, for each wave band we

    :param csp_x: csp transformed data (wave bands, features, samples)
    :param labels: list of labels for each sample (or numpy array).
    :return: a list of lists of models. Each list is a list containing a trained
    model of the same type for the each band.
    '''
    models = [[],[],[],[],[]]

    for ix, classifier in enumerate(classifier_types):
        for csp_ix, csp_exp in enumerate(csp_x):
            if classifier == SVC:
                inst = classifier(probability=True)
            elif classifier == KNeighborsClassifier:
                inst = KNeighborsClassifier(n_neighbors=9)
            else:
                inst = classifier()
            try:
                inst.fit(csp_exp, labels)
            except (AttributeError, ValueError):
                inst.fit(csp_exp, labels[csp_ix])
            models[ix].append(inst)

    return models

def trad_classifier_train(csp_x, labels, designed_classifier=None, classifier_types = (SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier, GaussianProcessClassifier)):
    '''
    Trains a list of list classifiers from the extracted features obtained from
    transforming the original FFT/EEG waves. So, for each wave band we

    :param csp_x: csp transformed data (wave bands, features, samples)
    :param labels: list of labels for each sample (or numpy array).
    :return: a list of lists of models. Each list is a list containing a trained
    model of the same type for the each band.
    '''

    accuracy_max = 0
    models = [[],[],[],[],[]]

    for ix, classifier in enumerate(classifier_types):
        for csp_ix, csp_exp in enumerate(csp_x):
            if classifier == SVC:
                inst = classifier(probability=True)
            elif classifier == KNeighborsClassifier:
                inst = KNeighborsClassifier(n_neighbors=9)
            else:
                inst = classifier()
            try:
                inst.fit(csp_exp, labels)
            except (AttributeError, ValueError):
                inst.fit(csp_exp, labels[csp_ix])

            models[ix].append(inst)

            try:
                aux = np.mean(np.equal(np.argmax(inst.predict_proba(csp_exp), axis=1), labels))
            except (AttributeError, ValueError):
                aux = np.mean(np.equal(np.argmax(inst.predict_proba(csp_exp), axis=1), labels[csp_ix]))
            if aux > accuracy_max:
                model_best = ix
                accuracy_max = aux

            if not (designed_classifier is None):
                if designed_classifier == classifier:
                    model_best = ix

    return models[model_best]

def fitted_classifiers_forward(X, models, clases):
    '''
    Takes the csp output and forwards a list of trained classifiers, returning
    a list of arrays of shape (classifiers, samples, clases).
    (For mulimodal decision, EMF framework)

    :param X: csp output. Array of (wave bands, samples, clases)
    :param classifiers: list of fitted classifiers. Must implement predict_proba() method.
    :param clases: number of different target clases.
    :return : an raay of shape (classifiers, samples, clases) thjat contains the logits
    for these classifiers.
    '''
    classifiers = []
    for model_set in models:
        logits_model = np.zeros((len(model_set), X[0].shape[0], clases))

        for ix, model in enumerate(model_set):
            logits_model[ix, :, :] = model.predict_proba(X[ix])

        classifiers.append(logits_model)

    return classifiers

def fitted_trad_classifier_forward(X, classifiers, clases):
    '''
    Takes the csp output and forwards a list of trained classifiers, returning
    an array of shape (classifiers, samples, clases)
    (For traditional decision making, trad framework)

    :param X: csp output. Array of (wave bands, samples, clases)
    :param classifiers: list of fitted classifiers. Must implement predict_proba() method.
    :param clases: number of different target clases.
    :return : an raay of shape (classifiers, samples, clases) thjat contains the logits
    for these classifiers.
    '''
    logits_model = np.zeros((len(classifiers), X[0].shape[0], clases))

    for ix, model in enumerate(classifiers):
        logits_model[ix, :, :] = model.predict_proba(X[ix])

    return logits_model

def multimodal_fusion_learn(logits_list, y):
    '''
    Learns the best pair of aggregation functions for a training data,
    using the best accuracy as criteria.

    :param logits: list of logits [(bands, samples, clases)] (x classifiers)
    :param y: labels for each sample.
    :return : (aggregation for phase 1, aggreation for phase 2)
    '''
    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values

    mejor = 0
    ag1 = agg_functions_names[0] #Precautionary default values
    ag2 = agg_functions_names[0]

    for ix, ag1 in enumerate(agg_functions_names):
        for jx, ag2 in enumerate(agg_functions_names):
            candidato = np.mean(np.argmax(ag1(ag2(final_logits,axis=0,keepdims=False), axis=0, keepdims=False), axis=1) == y)

            if candidato > mejor:
                mejor = candidato
                bag1 = ag1
                bag2 = ag2

    return bag1, bag2

def multimodal_fusion(logits, ag1, ag2, num_classes=None):
    '''
    Fuse a list of logits using ag1 and ag2 as aggregation functions.

    :param logits: list of logits [(bands, samples, clases)] (x classifiers)
    :param ag1: aggregation function.
    :param ag2: aggregation function.
    :param num_classes: different number of classes  (deprecated)
    '''
    res = np.zeros((len(logits), logits[0].shape[1],  logits[0].shape[-1]))
    for ix, freq_logits in enumerate(logits):
        res[ix, :, :] = ag1(freq_logits, axis=0, keepdims=False)

    return ag2(res, axis=0, keepdims=False)

def csp_forward(x, models):
    '''
    Given a training data of waves, and a list of csp models.

    :param X: train (bands, time, samples)
    :param models: list of csp models of len == bands.One csp model for each band.
    :return : (number of wave bands, features, samples)
    '''
    return lb.apply_CSP(x, models)

def csp_train_layer(X, y, n_filters=25):
    '''
    Trains a csp for each wave band in the load_brain_data module.

    :param X: train (bands, time, samples)
    :param y: labels (samples,)
    :param n_filters: max number of filters for each wave band. If the wave band
    has less frequencies than the number of filters, it takes the number of freqs
    as n_filters.
    :return : a tuple of (list of csp models, list of csp outputs)
    '''
    csp1, csp2, csp3, csp4, csp5, csp_senmot, y, csp_models = lb.calc_std_CSP(X, y, n_filters=n_filters)

    return csp_models, [csp1, csp2, csp3, csp4, csp5, csp_senmot]

# =============================================================================
# ARCHITECTURES
# =============================================================================
class bci_achitecture:
    '''
    This class is an abstraction of the bci architecture paradigm.
    Calling any of the methods appropietly will result
    in training and storing the subsenquently models.

    The way of generating a bci_architecture follows this procedure:
        > bci_arch = bci_architecture() <- This generates an "empty" bci architecture.
        > bci_arch.trad_architecture(X, y) <- This generates and trains all the components
                                                in a trad framework to perform classification
                                                in the X, y training set.
    Then, to perform classification no new samples, we can use two methods:
        > bci_arch.forward(X_test) <- Returns the predicted class

        or:

        > bci_arch.predict_proba(X_test) <- Returns the logits for each class.

    NOTE: bci architectures here presented do not perform signal preprocessing,
    the data is theoretically preprocessed in the load_brain_data module.
    '''

    def forward(self, *args):
            return np.argmax(self.predict_proba(*args), axis=1)


# =============================================================================
#     TRAD ARCHITECTURE
# =============================================================================
    def trad_architecture(self, X, y, verbose=False, agregacion=None):
        '''
        The trad architecture takes as features a set of different wave bands,
        in the original paper: the theta, delta, alpha, beta and All.

        For each one of them, we train a CSP model and a classifier. Then, we
        look for the best decission making function.

        We can specify the classifier used and the aggregation function used in the
        agregacion parameter. This might relax oversampling and save time, as
        looking for the best function might be very time  consuming.

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        :param verbose: print additional information (not supported as of today)
        :param agregacion: a tuple containing (agregation function, classifier)
                           objects. Can be none, in that case they will be learnt from the training set.
        '''
        def _trad_predict_proba(Xf, csp_models, classifiers, ag1, num_classes):
            csp_x = csp_forward(Xf, csp_models)
            logits = fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)

            return ag1(logits, axis=0, keepdims=False)
        try:
            if len(agregacion) == 2:
                agregacion, clasificador = agregacion
        except TypeError:
            clasificador = None
        self._num_classes = len(np.unique(y))

        csp_models, csp_x = csp_train_layer(X, y)

        classifier = trad_classifier_train(csp_x, y, designed_classifier=clasificador)

        logits = fitted_trad_classifier_forward(csp_x, classifier, self._num_classes)

        if agregacion is None:
            ag1 = decision_making_learn(logits, y)
        else:
            try:

                ag1 = bp.parse(agregacion)
            except AttributeError:
                ag1 = agregacion

        self.predict_proba = lambda a: _trad_predict_proba(a, csp_models, classifier, ag1, self._num_classes)

# =============================================================================
#          EMF ACHITECTURE
# =============================================================================
    def emf_architecture(self, X, y, verbose=False):
        '''
        The Enhanced Multimodal Framework architecture takes as features a set of different wave bands,
        in the original paper: the theta, delta, alpha, beta and All.

        For each one of them, we train a CSP model and a set of classifiers: KNN, LDA, QDA, SVM, GP.
        Then, we look for the best multimodal-decission making function.
        (It works similarly as an ensemble of trad frameworks)

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        :param verbose: print additional information (not supported as of today)
        '''
        def _emf_predict_proba(X, csp_models, classifiers, ag1, ag2, num_classes):
            csp_x = csp_forward(X, csp_models)
            logits = fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

            return multimodal_fusion(logits, ag1, ag2, num_classes)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = csp_train_layer(X, y)

        classifiers = classifier_std_block_train(csp_x, y)

        logits = fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

        ag1, ag2 = multimodal_fusion_learn(logits, y)

        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, ag1, ag2, self._num_classes)

# =============================================================================
# ALONE ARCHITECTURES
# =============================================================================
    def one_classifier_architecture(self, X, y, verbose=True, agregacion=0):
            '''
            The 1-classifier architecture takes as features a set of different wave bands: the theta, delta, alpha, beta and All.
            But will only use the All them (the rest are for compatibility reasons).

            For each one of them, we train a CSP model and a classifier. We select
            one classifier of the using the agregacion parameter.



            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            :param verbose: print additional information (not supported as of today)
            :param agregacion: an integer that designates the classifier to be used. Look for classifier_std_block_train()
                                doc to understand how they are numerated.

            '''

            def _emf_predict_proba(X, classifier, csp_models, classifiers):
                csp_x = csp_forward(X, csp_models)
                logits = fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

                return logits[classifier][4, :, :]

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = csp_train_layer(X, y)

            classifiers = classifier_std_block_train(csp_x, y)

            self.predict_proba = lambda a: _emf_predict_proba(a, agregacion, csp_models, classifiers)


if __name__ == '__main__':
    #A working example.
    from sklearn.model_selection import train_test_split
    import random
    random.seed(4)
    np.random.seed(4)

    #We load the data (remember to specify where it is)
    datasets_2a = lb.load_datasets_bci2a(derivate=2, normalize=True, tongue_feet=True)
    X, y = datasets_2a[0]
    X_train, X_test, y_train, y_test = train_test_split(np.transpose(X, (2, 1, 0)), y, test_size=0.5)
    X_train = np.transpose(X_train, (2, 1, 0))
    X_test = np.transpose(X_test, (2, 1, 0))

    #We create the bci architecture object
    my_architecture = bci_achitecture()
    #We transform this architecture in a emf one.
    my_architecture.emf_architecture(X_train, y_train)

    #We show our results
    print('Train accuracy: ' + str(np.mean(np.equal(my_architecture.forward(X_train), y_train))))
    print('Test accuracy: ' + str(np.mean(np.equal(my_architecture.forward(X_test), y_test))))