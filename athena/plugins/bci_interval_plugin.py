# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:50:40 2020

BCI plugin for the interval-valued moderate deviations and OWA operators.
Includes the basic functions to learn and apply the parameters and the
two architectures in the corresponding paper.

arXiv version:
    Fumanal-Idocin, J., Takáč, Z., Sanz, J. F. J. A., Goyena, H., Lin, C. T., Wang, Y. K., & Bustince, H. (2020).
    Interval-valued aggregation functions based on moderate deviations applied to Motor-Imagery-Based Brain Computer Interface.
    arXiv preprint arXiv:2011.09831.
@author: Javier Fumanal Idocin (Javier)
"""
import numpy as np

from Fancy_aggregations import moderate_deviations, implications, intervals, intervaluate_OWA
from Fancy_aggregations import binary_parser as bp
from Fancy_aggregations import intervaluate_moderate_deviations as md
from bci_architectures import weight_numerical_optimization, mff_aggregations

import bci_architectures as bci_base

def intervaluate_logits(logits, y=0.3, implication_operator=implications.reichenbach_implication):
    '''
    logits are of shape (waves, samples, clases) or a list of it.
    returns logits of shape (wave, samples, clases, intervalar) or a list of it.
    '''
    from scipy.stats.mstats import trim

    if isinstance(logits, list):
        res = []
        for logit in logits:
            res.append(trim(intervals.intervaluate(logit, y, implication_operator), (0,1)))
        return res
    else:
        return trim(intervals.intervaluate(logits, y, implication_operator), (0,1))

# =============================================================================
#  LEARN FUNCTIONS
# =============================================================================
def md_params_learn(X, y):
    '''
    Learns the parameters for a md function using a basic optimization process.
    (Look for basinhopping docs in scipy)

    :param X: training data. A numpy matrix of (features, samples, clases)
    :param y: labels

    '''
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(X, y, alpha):
        agg_logits = moderate_deviations.md_aggregation(X, axis=0, keepdims=False,  Mp=alpha[0], Mn=alpha[1])
        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    niter = 50
    function_alpha = lambda a: optimize_function(X, y, a)
    res = weight_numerical_optimization(function_alpha, [1, 10], niter=niter, mode='basinhopping')# minimizer_kwargs=minimizer_kwargs)

    return res

def multimodal_md_fusion_learn(logits_list, y):
    '''
    Performs the multimodal fusion learn process given a list with all the (features, samples, classes)
    base matrix using an intervalar md in both phases of the fusion.

   :param X: list of matrixes (features, samples, clases)
   :param y: labels (samples, )

    '''
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(logits, y, alpha):
        classifiers, waves, samples, clases = logits.shape
        alpha = alpha.reshape((classifiers+1,2))
        p1_logits = np.zeros((classifiers, samples, clases))

        for ix in range(classifiers):
            p1_logits[ix,:,:] = moderate_deviations.md_aggregation(logits[ix, :,:,:], axis=0, keepdims=False,  Mp=alpha[ix,0], Mn=alpha[ix,1])

        agg_logits = moderate_deviations.md_aggregation(p1_logits[:,ix,:,:], axis=0, keepdims=False, Mp=alpha[-1,0], Mn=alpha[-1,1])
        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values
    function_alpha = lambda a: optimize_function(final_logits, y, a)
    x0 = np.array([1, 10] * (len(logits_list)+1))
    res = weight_numerical_optimization(function_alpha, x0, bounds=[0, 100], niter=100, mode='basinhopping')

    return res

def multimodal_semi_md_fusion_learn(logits_list, y, layer=1):
    '''
    Performs the multimodal fusion learn process given a list with all the (features, samples, classes)
    base matrix. Uses an intervalar md in the designed layer and learns another
    fusion function for the other layer.

   :param X: list of matrixes (features, samples, clases)
   :param y: labels (samples, )

    '''
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(logits, y, ag, alpha, layer=1):
        classifiers, waves, samples, clases = logits.shape
        alpha = alpha.reshape((classifiers,2))
        p1_logits = np.zeros((classifiers, samples, clases))

        for ix in range(classifiers):
            if layer == 1:
                p1_logits[ix,:,:] = moderate_deviations.md_aggregation(logits[ix, :,:,:], axis=0, keepdims=False,  Mp=alpha[ix,0], Mn=alpha[ix,1])
            else:
                ag(logits[:,ix,:,:], axis=0, keepdims=False)

        if layer == 1:
            agg_logits = ag(logits[:,ix,:,:], axis=0, keepdims=False)
        else:
            agg_logits = moderate_deviations.md_aggregation(logits[:,ix,:,:], axis=0, keepdims=False,  Mp=alpha[ix,0], Mn=alpha[ix,1])

        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values

    mejor = 0.0
    for ag_name in mff_aggregations:
        function_alpha = lambda a: optimize_function(final_logits, y, bp.parse(ag_name), a)
        x0 = np.array([1, 10] * (len(logits_list)))
        res = weight_numerical_optimization(function_alpha, x0, bounds=[0, 100], niter=100, mode='basinhopping')
        candidato = function_alpha(res)

        if candidato > mejor:
                mejor = candidato
                bag = ag_name
                final_alpha = res

    return bag, final_alpha

def _compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

def owa_params_learn(X, y, owa_operator=intervaluate_OWA.iowa1, niter=50):

    def optimize_function(X, y, alpha):
        alpha_order, beta_order = alpha
        agg_logits = owa_operator(X, axis=0, keepdims=False)
        yhat = intervals.admissible_intervalued_array_argsort(agg_logits, axis=1, alpha_order=alpha_order, beta_order=beta_order)

        return 1 - _compute_accuracy(yhat, y)

    function_alpha = lambda a: optimize_function(X, y, a)
    res = weight_numerical_optimization(function_alpha, [0.5, 0.1], niter=niter, mode='montecarlo')

    return res

def multimodal_owa_fusion_learn(logits_list, y, owa_operator=intervaluate_OWA.iowa1, niter=50):

    def optimize_function(logits, y, alpha):
        classifiers, waves, samples, clases, interval_dim = logits.shape
        alpha_order1, beta_order1, alpha_order2, beta_order2 = alpha
        p1_logits = np.zeros((classifiers, samples, clases, interval_dim))

        for ix in range(classifiers):
            p1_logits[ix,:,:] = owa_operator(logits[ix, :,:,:], axis=0, keepdims=False,  alpha_order=alpha_order1, beta_order=beta_order1)

        agg_logits = owa_operator(p1_logits, axis=0, keepdims=False, alpha_order=alpha_order2, beta_order=beta_order2)
        yhat = intervals.admissible_intervalued_array_argsort(agg_logits, axis=1, alpha_order=alpha_order2, beta_order=beta_order2)

        return 1 - _compute_accuracy(yhat, y)

    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2], logits_list[0].shape[3]))

    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values

    function_alpha = lambda a: optimize_function(final_logits, y, a)
    x0 = [0.5, 0.1, 0.5, 0.1]
    res = weight_numerical_optimization(function_alpha, x0, niter=niter, mode='montecarlo')

    return res
# =============================================================================
# FORWARD FUNCTIONS
# =============================================================================
def multimodal_md_fusion(logits_list, alpha):
    '''
    Performs the multimodal fusion process given a list with all the (features, samples, classes)
    base matrix. Uses an intervalar md in both phases of the fusion.

   :param X: list of matrixes (features, samples, clases)
   :param y: labels (samples, )

    '''
    logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       logits[ix, :, :, :] = values

    classifiers, waves, samples, clases = logits.shape
    alpha = alpha.reshape((classifiers+1,2))
    p1_logits = np.zeros((classifiers, samples, clases))

    for ix in range(classifiers):
        p1_logits[ix,:,:] = moderate_deviations.md_aggregation(logits[ix, :,:,:], axis=0, keepdims=False,  Mp=alpha[ix,0], Mn=alpha[ix,1])

    return moderate_deviations.md_aggregation(logits[:,ix,:,:], axis=0, keepdims=False, Mp=alpha[-1,0], Mn=alpha[-1,1])

def multimodal_semi_md_fusion(logits_list, alpha, ag, layer=1):
    '''
    Performs the multimodal fusion process given a list with all the (features, samples, classes)
    base matrix. Uses an intervalar md in the designed layer and another
    fusion function for the other layer.

   :param X: list of matrixes (features, samples, clases)
   :param y: labels (samples, )

    '''
    if isinstance(ag, str):
        ag = bp.parse(ag)

    logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))

    for ix, values in enumerate(logits_list):
       logits[ix, :, :, :] = values

    classifiers, waves, samples, clases = logits.shape
    alpha = alpha.reshape((classifiers,2))
    p1_logits = np.zeros((classifiers, samples, clases))

    for ix in range(classifiers):
        if layer == 1:
            p1_logits[ix,:,:] = moderate_deviations.md_aggregation(logits[ix, :,:,:], axis=0, keepdims=False,  Mp=alpha[ix,0], Mn=alpha[ix,1])
        else:
            p1_logits[ix,:,:] = ag(logits[ix, :,:,:], axis=0, keepdims=False)

    if layer == 1:
        return ag(p1_logits[ix,:,:], axis=0, keepdims=False)
    else:
        return moderate_deviations.md_aggregation(p1_logits[ix,:,:], axis=0, keepdims=False, Mp=alpha[ix,0], Mn=alpha[ix,1])

def multimodal_owa_fusion(logits_list, alpha, owa_operator=intervaluate_OWA.iowa1):
    '''
    Performs the multimodal fusion process given a list with all the (features, samples, classes)
    base matrix. Uses an intervalar OWA in both phases of the fusion process.

   :param X: list of matrixes (features, samples, clases)
   :param y: labels (samples, )

    '''
    alpha_order1, beta_order1, alpha_order2, beta_order2 = alpha
    logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2], logits_list[0].shape[3]))

    for ix, values in enumerate(logits_list):
       logits[ix, :, :, :] = values

    classifiers, waves, samples, clases, intervalar_dim = logits.shape

    p1_logits = np.zeros((classifiers, samples, clases, intervalar_dim))

    for ix in range(classifiers):
        p1_logits[ix] = owa_operator(logits[ix, :,:,:], axis=0, keepdims=False,  alpha_order=alpha_order1, beta_order=beta_order1)

    return owa_operator(logits[:,ix,:,:,:], axis=0, keepdims=False, alpha_order=alpha_order2, beta_order=beta_order2)
# =============================================================================
#         INTERVALAR ARCHITECTURES
# =============================================================================
class bci_achitecture_interval_md(bci_base.bci_achitecture):

    def trad_intervalar_md_architecture(self, X, y, verbose=True, agregacion=None):
        '''
        Traditional BCI framework with interval-valued moderate aggregation techniques.

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes, intervalar_func=None, md_func=None):
            csp_x = bci_base.csp_forward(Xf, csp_models)
            logits = bci_base.fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)
            if intervalar_func is None:
                intervalar_logits = intervaluate_logits(logits)
            else:
                intervalar_logits = intervaluate_logits(logits, implication_operator=intervalar_func)

            if md_func is None:
                return md.md_interval_aggregation(intervalar_logits, axis=0, keepdims=False, alpha_order=ag1[0], beta_order=ag1[1], Mp=ag1[2], Mn=ag1[3])
            else:
                return md.md_interval_aggregation(intervalar_logits, axis=0, keepdims=False, alpha_order=ag1[0], beta_order=ag1[1], Mp=ag1[2], Mn=ag1[3], md=md_func)

        def _forward(Xf, csp_models, classifiers, ag1, num_classes,  intervalar_func=None, md_func=None):
             aggregated = _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes, intervalar_func, md_func)

             return intervals.admissible_intervalued_array_argsort(aggregated, alpha_order=ag1[0], beta_order=ag1[1], axis=1)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = bci_base.csp_train_layer(X, y)

        classifier = bci_base.trad_classifier_train(csp_x, y)

        logits = bci_base.fitted_trad_classifier_forward(csp_x, classifier, self._num_classes)

        if agregacion is None:
            intervalar_logits = intervaluate_logits(logits)
            ag1 = md_params_learn(intervalar_logits, y)
        else:
            interval_func, md_func = agregacion
            intervalar_logits = intervaluate_logits(logits, implication_operator=interval_func)
            ag1 = md_params_learn(intervalar_logits, y, md_func=md_func)

        if verbose:
            print(ag1)

        self._components = [csp_models, classifier, ag1, self._num_classes]
        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifier, ag1, self._num_classes)
        self.forward = lambda a: _forward(a, csp_models, classifier, ag1, self._num_classes)

    def emf_intervalar_md_architecture(self, X, y, verbose=True, agregacion=None):
        '''
        Enhanced Multimodal Fusion BCI framework with interval-valued moderate aggregation techniques.

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''

        def _emf_predict_proba(X, csp_models, classifiers, alpha, num_classes, intervalar_func=None, md_func=None):
            csp_x = bci_base.csp_forward(X, csp_models)
            logits = bci_base.fitted_classifiers_forward(csp_x, classifiers, self._num_classes)
            if intervalar_func is None:
                intervalar_logits = intervaluate_logits(logits)
            else:
                intervalar_logits = intervaluate_logits(logits, implication_operator=intervalar_func)

            if md_func is None:
                return multimodal_md_fusion(intervalar_logits, alpha)
            else:
                return multimodal_md_fusion(intervalar_logits, alpha, md_func=md_func)

        def _forward(Xf, csp_models, classifiers, ag1, num_classes, intervalar_func=None, md_func=None):
             aggregated = _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes, intervalar_func, md_func)

             return intervals.admissible_intervalued_array_argsort(aggregated, alpha_order=ag1[1][0], beta_order=ag1[1][1], axis=1)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = bci_base.csp_train_layer(X, y)

        classifiers = bci_base.classifier_std_block_train(csp_x, y)

        logits = bci_base.fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

        if agregacion is None:
            intervalar_logits = intervaluate_logits(logits)
            alpha = multimodal_md_fusion_learn(intervalar_logits, y)
        else:
            interval_func, md_func = agregacion
            intervalar_logits = intervaluate_logits(logits, implication_operator=interval_func)
            alpha = multimodal_md_fusion_learn(intervalar_logits, y,  md_func=md_func)


        if verbose:
            print(alpha)

        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, alpha, self._num_classes)
        self.forward = lambda a: _forward(a, csp_models, classifiers, alpha, self._num_classes)

    def trad_intervalar_owa_architecture(self, X, y, verbose=True, agregacion=None):
        '''
        Traditional BCI framework with interval-valued moderate aggregation techniques.

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes, intervaluate_func=None, owa_operator=None):
            csp_x = bci_base.csp_forward(Xf, csp_models)
            logits = bci_base.fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)
            if intervaluate_func is None:
                intervalar_logits = intervaluate_logits(logits)
            else:
                intervalar_logits = intervaluate_logits(logits, implication_operator=intervaluate_func)

            if owa_operator is None:
                return intervaluate_OWA.iowa1(intervalar_logits, axis=0, keepdims=False, alpha_order=ag1[0], beta_order=ag1[1])
            else:
                return owa_operator(intervalar_logits, axis=0, keepdims=False, alpha_order=ag1[0], beta_order=ag1[1])

        def _forward(Xf, csp_models, classifiers, ag1, num_classes, intervaluate_func=None, owa_operator=None):
             aggregated = _emf_predict_proba(Xf, csp_models, classifiers, ag1, num_classes)

             return intervals.admissible_intervalued_array_argsort(aggregated, alpha_order=ag1[0], beta_order=ag1[1], axis=1)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = bci_base.csp_train_layer(X, y)

        classifier = bci_base.trad_classifier_train(csp_x, y)

        logits = bci_base.fitted_trad_classifier_forward(csp_x, classifier, self._num_classes)


        if agregacion is None:
            intervalar_logits = intervaluate_logits(logits)
            ag1 = owa_params_learn(intervalar_logits, y)
        else:
            intervaluate_implication, owa_operator = agregacion
            intervalar_logits = intervaluate_logits(logits, implication_operator=intervaluate_implication)
            ag1 = owa_params_learn(intervalar_logits, y,  owa_operator=owa_operator)

        if verbose:
            print(ag1)

        self._components = [csp_models, classifier, ag1, self._num_classes]
        if agregacion is None:
            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifier, ag1, self._num_classes)
            self.forward = lambda a: _forward(a, csp_models, classifier, ag1, self._num_classes)
        else:
            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifier, ag1, self._num_classes, intervaluate_implication, owa_operator)
            self.forward = lambda a: _forward(a, csp_models, classifier, ag1, self._num_classes, intervaluate_implication, owa_operator)

    def emf_intervalar_owa_architecture(self, X, y, verbose=True, agregacion=None):
        '''
        Enhanced Multimodal Fusion BCI framework with interval-valued moderate aggregation techniques.

        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''

        def _emf_predict_proba(X, csp_models, classifiers, alpha, num_classes, intervaluate_func=None, owa_operator=None):
            csp_x = bci_base.csp_forward(X, csp_models)
            logits = bci_base.fitted_classifiers_forward(csp_x, classifiers, self._num_classes)
            if intervaluate_func is None:
                intervalar_logits = intervaluate_logits(logits)
            else:
                intervalar_logits = intervaluate_logits(logits, implication_operator=intervaluate_func)

            if owa_operator is None:
                return multimodal_owa_fusion(intervalar_logits, alpha)
            else:
                return multimodal_owa_fusion(intervalar_logits, alpha, owa_operator=owa_operator)

        def _forward(Xf, csp_models, classifiers, alpha, num_classes, intervaluate_func=None, owa_operator=None):
             aggregated = _emf_predict_proba(Xf, csp_models, classifiers, alpha, num_classes, intervaluate_func, owa_operator)

             return intervals.admissible_intervalued_array_argsort(aggregated, alpha_order=alpha[2], beta_order=alpha[3], axis=1)

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = bci_base.csp_train_layer(X, y)

        classifiers = bci_base.classifier_std_block_train(csp_x, y)

        logits = bci_base.fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

        if agregacion is None:
            intervalar_logits = intervaluate_logits(logits)
            alpha = multimodal_owa_fusion_learn(intervalar_logits, y)
        else:
            intervaluate_implication, owa_operator = agregacion
            intervalar_logits = intervaluate_logits(logits, implication_operator=intervaluate_implication)
            alpha = multimodal_owa_fusion_learn(intervalar_logits, y, owa_operator=owa_operator)

        if verbose:
            print(alpha)

        if agregacion is None:
            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, alpha, self._num_classes)
            self.forward = lambda a: _forward(a, csp_models, classifiers, alpha, self._num_classes)
        else:
            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, alpha, self._num_classes, intervaluate_implication, owa_operator)
            self.forward = lambda a: _forward(a, csp_models, classifiers, alpha, self._num_classes, intervaluate_implication, owa_operator)