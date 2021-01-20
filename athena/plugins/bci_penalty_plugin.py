# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:55:33 2020

@author: javi-
"""
import numpy as np

from Fancy_aggregations import penalties
from Fancy_aggregations import binary_parser as bp
from ..bci_architectures import classical_aggregations, mff_aggregations, _my_optimization

from .. import bci_architectures as athena

# =============================================================================
#  LEARN FUNCTIONS
# =============================================================================
def _penalty_cost_learn(X, y):
    from Fancy_aggregations import binary_parser
    from itertools import combinations

    classical_aggregations = ['mean', 'median', 'max', 'min']
    
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(X, y, cost_convex, alpha):
        alpha_cost = lambda real, yhat, axis: cost_convex(X, yhat, axis, alpha)
        
        agg_logits = penalties.penalty_aggregation(X, [binary_parser.parse(x) for x in classical_aggregations], axis=0, keepdims=False, cost=alpha_cost)
        yhat = np.argmax(agg_logits, axis=1)
        return 1 - compute_accuracy(yhat, y)
   
    max_value = 0
    
    cost_pool = penalties.base_cost_functions
    total_elements = len(cost_pool)
    r = 2
    cost_combs = combinations(range(total_elements), r)
    
    for cost1, cost2 in cost_combs:
        niter = 300
            
        cost = penalties._convex_comb(cost_pool[cost1], cost_pool[cost2])
        function_alpha = lambda a: optimize_function(X, y, cost, a)

        res = _my_optimization(function_alpha, niter=niter, mode='montecarlo')
        alpha_value = res

        if hasattr(alpha_value, 'len'):
            alpha_value = alpha_value[0]
        
        alpha_cost = lambda real, yhatf, axis: penalties._convex_comb(cost_pool[cost1], cost_pool[cost2])(real, yhatf, axis, alpha_value)

        agg_values = penalties.penalty_aggregation(X, [binary_parser.parse(x) for x in classical_aggregations], axis=0, keepdims=False, cost=alpha_cost)
        acc = compute_accuracy(np.argmax(agg_values, axis=1), y)
        if acc > max_value:
            max_value = acc
            cost_selected1 = cost1
            cost_selected2 = cost2
            alpha_final = alpha_value
                     
    for cost_ix in range(total_elements):
        agg_values = penalties.penalty_aggregation(X, [binary_parser.parse(x) for x in classical_aggregations], axis=0, keepdims=False, cost=cost_pool[cost_ix])
        acc = compute_accuracy(np.argmax(agg_values, axis=1), y)
        if acc > max_value:
            max_value = acc
            cost_selected1 = cost_ix
            cost_selected2 = cost_ix
            alpha_final = 1

    
    return lambda real, yhatf, axis: penalties._convex_comb(cost_pool[cost_selected1], cost_pool[cost_selected2])(real, yhatf, axis, alpha_final)

def _alpha_learn(X, y, cost):
    from Fancy_aggregations import binary_parser
    
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))
    
    def optimize_function(X, y, cost_convex, alpha):
        alpha_cost = lambda real, yhat, axis: cost_convex(X, yhat, axis, alpha)
        
        agg_logits = penalties.penalty_aggregation(X, [binary_parser.parse(x) for x in classical_aggregations], axis=0, keepdims=False, cost=alpha_cost)
        yhat = np.argmax(agg_logits, axis=1)
        
        return 1 - compute_accuracy(yhat, y)
   
    function_alpha = lambda a: optimize_function(X, y, cost, a)

    res = _my_optimization(function_alpha, niter=100, mode='montecarlo')
    alpha_value = res

    if hasattr(alpha_value, 'len'):
        alpha_value = alpha_value[0]
    
    return lambda real, yhatf, axis: cost(real, yhatf, axis, alpha_value)

def _multimodal_alpha_learn(X, y, cost, cost2):   
    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))
    
    def optimize_function(X, y, cost_convex, cost_convex2, alpha):
        alpha_cost1 = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha[0])
        alpha_cost2 = lambda real, yhat, axis: cost_convex2(real, yhat, axis, alpha[1])
        
        agg_logits = _multimodal_penalty_forward(X, alpha_cost1, alpha_cost2, len(np.unique(y)))
            
        yhat = np.argmax(agg_logits, axis=1)
        
        return 1 - compute_accuracy(yhat, y)
   
    function_alpha = lambda a: optimize_function(X, y, cost, cost2, a)

    res = _my_optimization(function_alpha, x0=[0.5, 0.5], niter=100, mode='montecarlo')
    alpha_value = res

    #if hasattr(alpha_value, 'len'):
    #    alpha_value = alpha_value[0]
    
    return lambda real, yhatf, axis: cost(real, yhatf, axis, alpha_value[0]), lambda real, yhatf, axis: cost(real, yhatf, axis, alpha_value[1])


def _multimodal_penalty_learn(logits_list, y, costs=None):
    from itertools import combinations
    
    agg_functions = [bp.parse(x) for x in mff_aggregations] #MFF set of aggregations
    
    def optimize_function(X, y, cost_convex1, cost_convex2, alphas):  
        alpha_cost1 = lambda real, yhat, axis: cost_convex1(real=real, yhat=yhat, axis=axis, alpha=alphas[0])
        alpha_cost2 = lambda real, yhat, axis: cost_convex2(real=real, yhat=yhat, axis=axis, alpha=alphas[1])
        
        freq_fusion = penalties.penalty_aggregation(X,axis=0,keepdims=False, agg_functions=agg_functions, cost=alpha_cost1)
        yhat = penalties.penalty_aggregation(freq_fusion, axis=0, keepdims=False,  agg_functions=agg_functions, cost=alpha_cost2)
        
        return 1 - np.mean(np.equal(np.argmax(yhat,axis=1), y))
    
    #agg_functions = [bp.parse(x) for x in agg_functions_names]
        
    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))
    
    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values
    
    mejor = 0
    X = final_logits
    
    cost_pool = penalties.base_cost_functions
    
    total_elements = len(cost_pool)
    r = 2
    cost_combs = combinations(range(total_elements), r)
    
    if costs is not None:
        function_alpha = lambda a: optimize_function(X, y, costs[0], costs[1], a)
        res = _my_optimization(function_alpha, x0=[0.5, 0.5], niter=100, mode='montecarlo')
        
        return lambda real, yhat, axis: costs[0](real, yhat, axis, res[0]),  lambda real, yhat, axis: costs[1](real, yhat, axis, res[1])
    
    for cost1a, cost1b in cost_combs:
        cost1 = penalties._convex_comb(cost_pool[cost1a], cost_pool[cost1b])
        
        cost_combs2 = combinations(range(total_elements), r)
        for cost2a, cost2b in cost_combs2:
            cost2 = penalties._convex_comb(cost_pool[cost2a], cost_pool[cost2b])
            function_alpha = lambda a: optimize_function(X, y, cost1, cost2, a)
            res = _my_optimization(function_alpha, x0=[0.5, 0.5], niter=100)#(function_alpha, [0.5, 0.5], minimizer_kwargs=minimizer_kwargs, niter=5)
            
            alpha_value = res
            alpha_cost1 = lambda real, yhat, axis: cost1(real, yhat, axis, alpha_value[0])
            alpha_cost2 = lambda real, yhat, axis: cost2(real, yhat, axis, alpha_value[1])
            
            freq_fusion = penalties.penalty_aggregation(X, axis=0,keepdims=False, agg_functions=agg_functions, cost=alpha_cost1)
            yhat = penalties.penalty_aggregation(freq_fusion, axis=0, keepdims=False,  agg_functions=agg_functions, cost=alpha_cost2)
            candidato = np.mean(np.equal(np.argmax(yhat,axis=1), y))

            if candidato > mejor: 
                mejor = candidato
                alpha_value1 = alpha_value[0]
                alpha_value2 = alpha_value[1]
                fcost1a = cost1a
                fcost1b = cost1b
                fcost2a = cost2a
                fcost2b = cost2b
                
    return lambda real, yhat, axis: penalties._convex_comb(cost_pool[fcost1a], cost_pool[fcost1b])(real, yhat, axis, alpha_value1),  lambda real, yhat, axis: penalties._convex_comb(cost_pool[fcost2a], cost_pool[fcost2b])(real, yhat, axis, alpha_value2)

def _multimodal_semi_penalty_learn(logits_list, y, layer=1, cost=None):
    #from sklearn.model_selection import train_test_split
    from itertools import combinations
    
    def optimize_function(X, y, cost_convex1, ag, alpha, layer=1):  
        alpha_cost1 = lambda real, yhat, axis: cost_convex1(real=real, yhat=yhat, axis=axis, alpha=alpha)
        
        if layer == 1:
            freq_fusion = penalties.penalty_aggregation(X,axis=0,keepdims=False, agg_functions=agg_functions, cost=alpha_cost1)
            yhat = ag(freq_fusion, axis=0, keepdims=False)
        else:
            freq_fusion = ag(X,axis=0,keepdims=False)
            yhat = penalties.penalty_aggregation(freq_fusion, axis=0, keepdims=False,  agg_functions=agg_functions, cost=alpha_cost1)    
            
        return 1 - np.mean(np.equal(np.argmax(yhat,axis=1), y))
    
    agg_functions = [bp.parse(x) for x in mff_aggregations]
        
    final_logits = np.zeros((len(logits_list), logits_list[0].shape[0], logits_list[0].shape[1], logits_list[0].shape[2]))
    
    for ix, values in enumerate(logits_list):
       final_logits[ix, :, :, :] = values
    
    mejor = 0
    
    #X_train, X_test, y_train, y_test = train_test_split(np.transpose(final_logits, (2, 0, 1, 3)), y, test_size=0.20, random_state=None)
    #X_train = np.transpose(X_train, (1, 2, 0, 3))
    #X_test = np.transpose(X_test, (1, 2, 0, 3))
    X_train = final_logits
    X_test = final_logits
    y_train = y
    y_test = y
    
    total_elements = len(penalties.base_cost_functions)
    r = 2
    cost_combs = combinations(range(total_elements), r)
    
    if cost is None:
        for cost1, cost2 in cost_combs:
            cost = penalties._convex_comb(penalties.cost_functions[cost1], penalties.cost_functions[cost2])
            
            for ag_name in mff_aggregations:
                ag = bp.parse(ag_name)
                function_alpha = lambda a: optimize_function(X_train, y_train, cost, ag, a, layer)
                res = _my_optimization(function_alpha, x0=[0.5], niter=200)
                alpha_cost1 = lambda real, yhat, axis: cost(real, yhat, axis, res)
                
                if layer == 1:
                    freq_fusion = penalties.penalty_aggregation(X_test, axis=0,keepdims=False, agg_functions=agg_functions, cost=alpha_cost1)
                    yhat = ag(freq_fusion, axis=0, keepdims=False)
                else:
                    freq_fusion = ag(X_test, axis=0,keepdims=False)
                    yhat = penalties.penalty_aggregation(freq_fusion, axis=0, keepdims=False,  agg_functions=agg_functions, cost=alpha_cost1)
                    
                candidato = np.mean(np.equal(np.argmax(yhat,axis=1), y_test))
    
                if candidato > mejor: 
                    mejor = candidato
                    bag1 = ag_name
                    fcost1 = cost1
                    fcost2 = cost2
                    final_alpha = res
                    
        return bp.parse(bag1),  lambda real, yhat, axis: penalties._convex_comb(penalties.cost_functions[fcost1], penalties.cost_functions[fcost2])(real, yhat, axis, final_alpha)
    else:
        for ag_name in mff_aggregations:
            ag = bp.parse(ag_name)
            
            
            if layer == 1:
                freq_fusion = penalties.penalty_aggregation(X_test, axis=0,keepdims=False, agg_functions=agg_functions, cost=cost)
                yhat = ag(freq_fusion, axis=0, keepdims=False)
            else:
                freq_fusion = ag(X_test, axis=0,keepdims=False)
                yhat = penalties.penalty_aggregation(freq_fusion, axis=0, keepdims=False,  agg_functions=agg_functions, cost=cost)
                
            candidato = np.mean(np.equal(np.argmax(yhat,axis=1), y_test))

            if candidato > mejor: 
                mejor = candidato
                bag1 = ag_name
                    
        return bp.parse(bag1),  cost

# =============================================================================
# FORWARD FUNCTIONS
# =============================================================================
def _forward_penalty(X, cost):
    from Fancy_aggregations import binary_parser
    
    agg_functions = [binary_parser.parse(x) for x in classical_aggregations]
    
    return penalties.penalty_aggregation(X, agg_functions, axis=0, keepdims=False, cost=cost)

def _multimodal_penalty_forward(logits, cost1, cost2, num_classes):
    from Fancy_aggregations import binary_parser
    
    res = np.zeros((len(logits), logits[0].shape[1], num_classes))
    for ix, freq_logits in enumerate(logits):
        res[ix, :, :] = penalties.penalty_aggregation(freq_logits, [binary_parser.parse(x) for x in mff_aggregations], axis=0, keepdims=False, cost=cost1)
    
    return penalties.penalty_aggregation(res, [binary_parser.parse(x) for x in mff_aggregations], axis=0, keepdims=False, cost=cost2)

def _multimodal_semi_penalty_forward(logits, ag, cost1, num_classes, layer=1):
    from Fancy_aggregations import binary_parser
    
    res = np.zeros((len(logits), logits[0].shape[1], num_classes))
    for ix, freq_logits in enumerate(logits):
        if layer == 1:
            res[ix, :, :] = penalties.penalty_aggregation(freq_logits, [binary_parser.parse(x) for x in mff_aggregations], axis=0, keepdims=False, cost=cost1)
        else:
            res[ix, :, :] = ag(freq_logits, axis=0, keepdims=False)
    
    if layer == 1:
        return ag(res, axis=0, keepdims=False)
    else:
        return penalties.penalty_aggregation(res, [binary_parser.parse(x) for x in mff_aggregations], axis=0, keepdims=False, cost=cost1)

class bci_achitecture_penalty(athena.bci_achitecture):

    # =============================================================================
    #          PENALTY ARCHITECTURES
    # =============================================================================
    def emf_penalty_architecture(self, X, y, verbose=False, costs = None):
            '''
            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            '''
        def _emf_predict_proba(X, csp_models, classifiers, ag1, ag2, num_classes):
                csp_x = _csp_forward(X, csp_models)
                logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

                return _multimodal_penalty_forward(logits, ag1, ag2, num_classes)

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = _csp_train_layer(X, y)

            classifiers = _classifier_std_block_train(csp_x, y)

            logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

            if costs is None:
                cost1, cost2 = _multimodal_penalty_learn(logits, y, costs)
            else:
                if hasattr(costs, '__len__'):
                    cost1 = costs[0]
                    cost2 = costs[1]
                else:
                    cost1 = costs
                    cost2 = costs
                import inspect
                if ('alpha0' in inspect.getfullargspec(cost1)[0]) or ('alpha' in inspect.getfullargspec(cost1)[0]):
                    cost1, cost2 = _multimodal_alpha_learn(logits, y, cost1, cost2)

            if verbose:
                print(cost1, cost2)

            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, cost1, cost2, self._num_classes)

    def emf_semi_penalty_architecture(self, X, y, verbose=True, opt=[1]):
            '''
            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            '''
            from bci_penalty_plugin import _multimodal_semi_penalty_learn, _multimodal_semi_penalty_forward
            def _emf_predict_proba(X, csp_models, classifiers, ag1, ag2, num_classes, layer):
                csp_x = _csp_forward(X, csp_models)
                logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

                return _multimodal_semi_penalty_forward(logits, ag1, ag2, num_classes, layer)

            if len(opt) == 1:
                layer = opt[0]
                cost = None
            else:
                layer = opt[0]
                cost = opt[1]

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = _csp_train_layer(X, y)

            classifiers = _classifier_std_block_train(csp_x, y)

            logits = _fitted_classifiers_forward(csp_x, classifiers, self._num_classes)

            cost1, cost2 = _multimodal_semi_penalty_learn(logits, y, layer, cost)

            if verbose:
                print(cost1, cost2)

            self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, cost1, cost2, self._num_classes, layer)


    def trad_penalty_architecture(self, X, y, verbose=True, cost=None):
            '''
            :param X: train (bands, time, samples)
            :param y: labels (samples,)
            '''
            from bci_penalty_plugin import _penalty_cost_learn, _forward_penalty, _alpha_learn

            def _trad_predict_proba(Xf, csp_models, classifiers, penalty_cost, num_classes):
                csp_x = _csp_forward(Xf, csp_models)
                logits = _fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)

                return _forward_penalty(logits, cost=penalty_cost)

            self._num_classes = len(np.unique(y))

            csp_models, csp_x = _csp_train_layer(X, y)

            classifier = _trad_classifier_train(csp_x, y, classifier_types=(DecisionTreeClassifier))

            logits = _fitted_trad_classifier_forward(csp_x, classifier, self._num_classes)

            if cost is None:
                cost = _penalty_cost_learn(logits, y)
            else:
                import inspect
                if ('alpha0' in inspect.getfullargspec(cost)[0]) or ('alpha' in inspect.getfullargspec(cost)[0]):
                    cost = _alpha_learn(logits, y, cost)
            self._components = [csp_models, classifier, cost, self._num_classes]

            if verbose:
                print(cost)

            self.predict_proba = lambda a: _trad_predict_proba(a, csp_models, classifier, cost, self._num_classes)    

