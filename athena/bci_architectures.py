# -*- coding: utf-8 -*-

"""
Functions to load architectures (frameworks) from the BCI experiment.

Created on Thu Mar 28 15:26:53 2019

@author: javi-
"""
import numpy as np

import load_brain_data as lb
import Fancy_aggregations.binary_parser as bp

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

classifier_types = [SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier, GaussianProcessClassifier]

agg_functions_names = ['sugeno', 'shamacher', 'choquet', 'cfminmin', 'cf12', 'cf', 'owa1', 'owa2', 'owa3', 'geomean', 'sinoverlap',
                'hamacher', 'luka', 'probabilistic_sum', 'bounded_sum', 'einstein_sum']

mff_aggregations = ['choquet', 'sugeno', 'shamacher', 'cfminmin']
classical_aggregations = ['mean', 'min', 'max', 'median']


def _fast_montecarlo_optimization(function_alpha, x0=[0.5], minimizer_kwargs=None, niter=200, smart=True):
    '''
    Just randomly samples the function. More functionality might come in the future if necessary.
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

def _fast_montecarlo_optimization_md(function_alpha, x0=[0.5], minimizer_kwargs=None, niter=200):
    '''
    Just randomly samples the function. More functionality might come in the future if necessary.
    Considering moving this function to the more fitting bci md plugin.
    '''
    class dummy_plug:
        def _init(self, x=None):
            self.x = x

    def gen_subject():
        alpha = np.random.random()
        beta = np.random.random()

        Mp = np.random.randint(50)
        Mn = np.random.randint(50)

        return [alpha, beta, Mp, Mn]

    iter_actual = 0

    #epsilon = 0.05
    eval_per_iter = 5
    best_fitness = 1
    resultado = dummy_plug()

    while(iter_actual < niter):
        if len(x0) == 2:
            subjects = [[gen_subject(), gen_subject()] for x in range(eval_per_iter)]
        else:
            subjects = [gen_subject() for x in range(eval_per_iter)]

        fitness = [function_alpha(x) for x in subjects]
        ordered = np.sort(fitness)
        arg_ordered = np.argsort(fitness)
        iter_actual += 1

        if ordered[1] < best_fitness:
            best_fitness = ordered[1]
            resultado.x = subjects[arg_ordered[1]]
            resultado.fun = best_fitness
            if best_fitness == 0.0:
                return resultado

    return resultado

def _my_optimization(func, x0=0.5, bounds=[0, 1], niter=100, mode='montecarlo', verbose=False):
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
    elif mode == 'montecarlo_md':
        res = _fast_montecarlo_optimization_md(func, x0=x0, niter=niter)
    elif mode == 'montecarlo':
        res = _fast_montecarlo_optimization(func, x0=x0, niter=niter)
    end = time.time()

    if verbose:
        print('Result: ' + str(res.x), 'Time taken: ' + str(end - start), 'Error: ' + str(res.fun) + '(Acc: ' + str(1 - res.fun) + ')')

    return res.x
# =============================================================================
#   BASIC COMPONENTS
# =============================================================================
def _extract_logits(X, machines, num_classes):
    '''
    Extract logits from a series of boltzmann models.

    :param X: (bands, time, samples)
    '''
    bands, time, samples = X.shape
    logits = np.zeros((len(machines), X.shape[2], num_classes))

    for ix, machine in enumerate(machines):
        waveband = lb.wavelets[ix]
        X_bands = X[waveband, :, :]
        X_reshaped = X_bands.reshape((len(waveband)*time, samples)).T
        logits[ix, :, :] = machine.predict_proba(X_reshaped>0)

    return logits


def _decision_making_learn(X, y):
    '''
    X (classifiers, samples, clases)
    '''
    from Fancy_aggregations import binary_parser as bp
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



def _classifier_std_block_train(csp_x, labels):
    '''

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

def _trad_classifier_train(csp_x, labels, designed_classifier=None, classifier_types = (SVC, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, KNeighborsClassifier, GaussianProcessClassifier)):
    '''

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

def _fitted_classifiers_forward(X, models, clases):
    classifiers = []
    for model_set in models:
        logits_model = np.zeros((len(model_set), X[0].shape[0], clases))

        for ix, model in enumerate(model_set):
            logits_model[ix, :, :] = model.predict_proba(X[ix])

        classifiers.append(logits_model)

    return classifiers

def _fitted_trad_classifier_forward(X, classifiers, clases):
    '''
    '''
    logits_model = np.zeros((len(classifiers), X[0].shape[0], clases))

    for ix, model in enumerate(classifiers):
        logits_model[ix, :, :] = model.predict_proba(X[ix])

    return logits_model

def _multimodal_fusion_learn(logits_list, y):
    '''
    '''
    from Fancy_aggregations import binary_parser as bp


    agg_functions1 = [bp.parse(x) for x in ['choquet', 'sugeno', 'shamacher', 'cfminmin', 'cf12', 'cf', 'fhamacher']]
    agg_functions2 = [bp.parse(x) for x in ['geomean', 'hmean', 'sinoverlap']]

    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values

    mejor = 0
    ag1 = agg_functions1[0] #Precautionary default values
    ag2 = agg_functions2[0]

    for ix, ag1 in enumerate(agg_functions_names):
        for jx, ag2 in enumerate(agg_functions_names):
            candidato = np.mean(np.argmax(ag1(ag2(final_logits,axis=0,keepdims=False), axis=0, keepdims=False), axis=1) == y)

            if candidato > mejor:
                mejor = candidato
                bag1 = ag1
                bag2 = ag2

    return bag1, bag2

def _multimodal_fusion(logits, ag1, ag2, num_classes):
    res = np.zeros((len(logits), logits[0].shape[1], num_classes))
    for ix, freq_logits in enumerate(logits):
        res[ix, :, :] = ag1(freq_logits, axis=0, keepdims=False)

    return ag2(res, axis=0, keepdims=False)

def _csp_forward(x, models):
    return lb.apply_CSP(x, models)

def _csp_train_layer(X, y, n_filters=25):
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
    '''

    def forward(self, *args):
            return np.argmax(self.predict_proba(*args), axis=1)



# =============================================================================
#     TRAD ARCHITECTURE
# =============================================================================
    def trad_architecture(self, X, y, verbose=False, agregacion=None):
        '''
        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes):
            csp_x = _csp_forward(Xf, csp_models)
            logits = _fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)

            return ag1(logits, axis=0, keepdims=False)
        try:
            if len(agregacion) == 2:
                agregacion, clasificador = agregacion
        except TypeError:
            clasificador = None
        self._num_classes = len(np.unique(y))

        csp_models, csp_x = _csp_train_layer(X, y)

        classifier = _trad_classifier_train(csp_x, y, designed_classifier=clasificador)

        logits = _fitted_trad_classifier_forward(csp_x, classifier, self._num_classes)

        if agregacion is None:
            ag1 = _decision_making_learn(logits, y)
        else:
            try:

                ag1 = bp.parse(agregacion)
            except AttributeError:
                ag1 = agregacion

        self._components = [csp_models, classifier, ag1, self._num_classes]
        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifier, ag1, self._num_classes)

# =============================================================================
#          EMF ACHITECTURE
# =============================================================================
    def emf_architecture(self, X, y, verbose=False):
        '''
        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _emf_predict_proba(X, csp_models, classifiers, ag1, ag2, num_classes):
            csp_x = _csp_forward(X, csp_models)
            logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

            return _multimodal_fusion(logits, ag1, ag2, num_classes)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = _csp_train_layer(X, y)

        classifiers = _classifier_std_block_train(csp_x, y)

        logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

        ag1, ag2 = _multimodal_fusion_learn(logits, y)

        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, ag1, ag2, self._num_classes)

# =============================================================================
# ALONE ARCHITECTURES
# =============================================================================
    def one_classifier_architecture(self, X, y, verbose=True, agregacion=0):
            '''
            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            '''

            def _emf_predict_proba(X, classifier, csp_models, classifiers):
                csp_x = _csp_forward(X, csp_models)
                logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

                return logits[classifier][4, :, :]

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = _csp_train_layer(X, y)

            classifiers = _classifier_std_block_train(csp_x, y)

            self.predict_proba = lambda a: _emf_predict_proba(a, agregacion, csp_models, classifiers)

    def csp_classifier_architecture(self, X, y, verbose=True):
            '''
            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            '''

            def _emf_predict_proba(X, csp_models):
                csp_x = _csp_forward(X, csp_models)

                return csp_x[4]

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = _csp_train_layer(X, y, n_filters=2)

            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import random
    random.seed(4)
    np.random.seed(4)

    datasets_2a = lb.load_datasets_bci2a(derivate=0, normalize=True, tongue_feet=True)
    my_architecture = bci_achitecture()
    X, y = datasets_2a[0]
    X_train, X_test, y_train, y_test = train_test_split(np.transpose(X, (2, 1, 0)), y, test_size=0.5)
    X_train = np.transpose(X_train, (2, 1, 0))
    X_test = np.transpose(X_test, (2, 1, 0))

    #my_architecture.trad_penalty_architecture(X_train, y_train, costs=[penalties.cost_functions[0], penalties.cost_functions[1]])
    my_architecture.mff_intervalar_owa_architecture(X_train, y_train)#, cost=penalties.cost_functions[0])

    print('Acierto de train: ' + str(np.mean(np.equal(my_architecture.forward(X_train), y_train))))
    print('Acierto de test: ' + str(np.mean(np.equal(my_architecture.forward(X_test), y_test))))