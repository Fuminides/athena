# -*- coding: utf-8 -*-
"""
Functions to make classifiers from CSP-result data.

Created on 16/05/2019

@author: Javier Fumanal Idocin
"""
import numpy as np

import load_brain_data as lb

def train_classifier_csp(csp_x, labels, classifier, ensemble_type = None, n_experts=60):
    '''
    Given a CSP-formated data and a set of labels, returns
    the trained set of models for each csp.

    :param csp_data: csp data from a series of experiments. (A tuple)
    :param classifier: scipy-like classifier.
    :return: The set of trained models.
    '''
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

    models = []

    for csp_exp in csp_x:
        if classifier == SVC:
            inst = classifier(probability=True)
        elif classifier == KNeighborsClassifier:
            inst = KNeighborsClassifier(n_neighbors=9)
        else:
            inst = classifier()

        if ensemble_type == 'boost':
            try:
                ada_classifiers = AdaBoostClassifier(inst, n_estimators=n_experts)
                ada_classifiers.fit(csp_exp, labels)
                for ada_i in ada_classifiers.estimators_:
                    models.append(ada_i)
            except ValueError:
                inst.fit(csp_exp, labels)
                models.append(inst)

        elif ensemble_type == 'bagging':
            boost_classifiers = BaggingClassifier(inst, n_estimators=n_experts, max_samples=0.5)
            boost_classifiers.fit(csp_exp, labels)
            for boost_i in boost_classifiers.estimators_:
                models.append(boost_i)

        elif ensemble_type == 'krypteia':
            import historic_computing.krypteia as kr

            spartan_classifiers, unit_biases, _ = kr.generate_unit(csp_exp, labels, classifier_list=[classifier], n=n_experts*2, ordeal_difficulty=0.8)
            spartan_unit_selected = kr.sexual_selection(spartan_classifiers, unit_biases, csp_exp, labels, pressure=0.05, selective=n_experts)
            for spartan in spartan_unit_selected:
                models.append(spartan)
        else: #elif 'simple'
            inst.fit(csp_exp, labels)
            models.append(inst)


    return models

def generate_logits(classifiers, X):
    '''

    :param classifiers: fitted clasifiers. They must have the "._predict_proba" method.

    :param X: X is a list with each n-th test set for each n-th classifier
    :return: the logits for that classifier.
    '''
    number_classifiers = len(classifiers)
    shape_logits = classifiers[0].predict_proba(X[0]).shape
    matrix_logits = np.zeros([number_classifiers, shape_logits[0], shape_logits[1]])
    ix = 0
    for n, fold in enumerate(X):
        for _, clasificador in enumerate(classifiers):
            try:
                logits = clasificador.predict_proba(fold)
                matrix_logits[ix, :, :] = logits
                ix += 1
            except Exception: #If dimensions doesnt hold up well
                pass
    return matrix_logits

def ensemble_decision(logits, decision_f = np.mean):
    '''
    Returns predictions for each sample.

    :param logits: matrix of logits for each class obtained from a list of classifiers.
    :param decision_f: the decision function used to perform the decision.
    :return: the final classification for the ensemble.
    '''
    agg_values = decision_f(logits, axis = 0)
    return np.argmax(agg_values, axis=len(agg_values.shape)-1)

def soft_ensemble_decision(logits, decision_f = np.mean):
    '''
    Returns puntuation for each sample. (Not predictions)

    :param logits: matrix of logits for each class obtained from a list of classifiers.
    :param decision_f: the decision function used to perform the decision.
    :return: the final probabilites for the ensemble.
    '''
    return decision_f(logits, axis = 0, keepdims = True)

def calculate_itr(p, t=2.911950945854187, n=2, N=178):
    '''
    Computes the ITR given the accuracy, the number of classes and the processing time.

    :param t: t is the time for processing the batch. Must be given in seconds.
    '''
    return (np.log2(n) + p*np.log2(p) + (1-p)*np.log2((1-p)/(n-1))) * 60 / t * N

def evaluate(yhat, y):
    '''
    Calculates the accuracy for a set of labels and corresponding ground truth.
    More success criteria may be added later, so use this function instead of manually calculating.

    :param yhat: predicted values.
    :param y: existing values.
    :return: accuracy
    '''
    accuracy = np.mean(yhat==y)

    return accuracy

def cross_validation_partitions(X, dataset='uts'):
    '''
    Generates a set of 5 folds to perform k-fold. It uses the 1st dimension to split the data.

    :param X: sample data.
    :return: a generator containing the k-fold indexes to perform cross_validation
    '''
    import random
    from sklearn.model_selection import RepeatedKFold, KFold
    random.seed(4) #My favourite number

    if dataset == 'uts':
        kf = KFold(n_splits=5, random_state=4)
    elif dataset == '2a':
        kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=4)

    return  kf.split(X)


def full_fold_eval(X, labels, classifier_type, logits2pred = ensemble_decision):
    '''
    A do-everything method: given the train data and labels, and a classifier class
    returns the accuracies obtained in a 5 k-fold evaluation.

    :param X: train data.
    :param labels: labels for each X sample.
    :return: accuracies for a 5-fold evaluation.
    '''
    accuracies = [0]*5
    ix = 0
    partitions = cross_validation_partitions(np.transpose(X, (2, 1, 0)), nfolds=10)

    for train, test in partitions:
        x, x_test, y, y_test = obtain_partitions(X, labels, test, train)

        csp_models, trained_models = fit_models(classifier_type, x, y)

        logits = extract_logits(csp_models, trained_models, x_test)

        yhat = logits2pred(logits)
        accuracies[ix] = evaluate(yhat, y_test)
        ix += 1

    return accuracies

def full_logits(X, labels, classifier_type, dataset='2a', deep_mode=2):
    '''

    :param X: train data.
    :param labels: labels for each X sample.
    :param deep_mode: 0 is not machine belief used. 1 is after the signal diff.
    2 is after the CSP. 3 is for both after the signal diff and and the CSP.
    :return: logits for a 5-fold evaluation.
    '''
    logits = []
    partitions = cross_validation_partitions(np.transpose(X, (2, 1, 0)), dataset=dataset)
    csp_models = None
    for train, test in partitions:
        x, x_test, y, y_test = obtain_partitions(X, labels, test, train)
        
        if (deep_mode == 1) or (deep_mode == 3):
            freqs, tiempo, pruebas = x.shape
            test_freqs, test_tiempo, test_pruebas = x_test.shape
            
            x = x.reshape((freqs, tiempo*pruebas)).T
            x_test = x_test.reshape((freqs, test_tiempo*test_pruebas)).T
            x, machine = deep_belief_transform(x)
            x_test = machine.transform(x_test)
            
            x = np.array(x).reshape((x.shape[1], tiempo, pruebas))
            x_test = np.array(x_test).reshape((x_test.shape[1], tiempo, test_pruebas))
            
        if csp_models is None:
            csp_models, trained_models = fit_models(classifier_type, x, y)
        else:
            _, trained_models = fit_models(classifier_type, x, y, csp_models)

        extracted_logits = extract_logits(csp_models, trained_models, x_test)
        
        logits.append((extracted_logits, y_test))

    if (deep_mode == 2) or (deep_mode == 3):
        return logits
    else:
        return logits

def evaluate_decision_making(logits, decision_function, waves=None):
    '''

    :param logits:
    :param labels:
    :param decision_function:
    :return:
    '''
    from inspect import getfullargspec
    accuracies = [0]*len(logits)
    if waves is None:
        waves = np.arange(logits[0][0].shape[0])

    for i in range(len(logits)):
        #Lucrezia decisor needs the labels so I made this monkey patch (Sorry)
        if 'labels' in getfullargspec(decision_function)[0]:
            yhat = decision_function(logits[i][0][waves, :, :], labels=logits[i][1])
        else:
            yhat = decision_function(logits[i][0][waves, :, :])

        accuracies[i] = evaluate(yhat, logits[i][1])

    return accuracies

def extract_logits(csp_models, trained_models, x_test):
    '''
    Calculate the logits for each observation.

    :param csp_models: csp calculated from the train data.
    :param trained_models: fitted models from the train data.
    :param x_test: data to predict.
    :return: the logits from each observation and classifier in a numpy array.
    '''
    test_csp = lb.apply_CSP(x_test, csp_models)
    logits = generate_logits(trained_models, test_csp)

    return logits


def fit_models(classifier_type, x, y, models=None):
    '''
    Fits the models to a wave data.



    :param classifier_type: A classifier.
    :param x: WxNxM wave.
    :param y: Labels
    :return: csp transformation and fitted models.
    '''
    if models is None:
        csp1, csp2, csp3, csp4, csp5, csp_senmot, y, csp_models = lb.calc_std_CSP(x, y)
    else:
        csp1, csp2, csp3, csp4, csp5, csp_senmot = lb.apply_CSP(x, models)
        csp_models = models

    trained_models = train_classifier_csp([csp1, csp2, csp3, csp4, csp5, csp_senmot], y, classifier_type)

    return csp_models, trained_models


def obtain_partitions(X, labels, test, train):
    '''
    Given a list of indexes and train data/labels obtains the train and test partition.

    :param X: observations
    :param labels: X labels
    :param test: list of test indexes.
    :param train:  list of train indexes.
    :return: x, x test, labels, labels test
    '''
    x = X[:, :, train]
    y = labels[train]
    x_test = X[:, :, test]
    y_test = labels[test]

    return x, x_test, y, y_test

def jaccard_distances(logits, format = 'plot'):
    '''

    :param logits:
    :return:
    '''
    nclasificadores, muestras, clases = logits.shape
    logits = np.argmax(logits, axis = 2)
    if format == 'matrix':
        resultados = np.zeros((nclasificadores, nclasificadores))
    elif format == 'plot':
        resultados = []

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            if format == 'matrix':
                resultados[i,j] = np.mean(logits[i, :] == logits[j, :])
            elif format == 'plot':
                resultados.append((i, j, np.mean(logits[i, :] == logits[j, :])))

    return resultados

def positive_jaccard_distances(logits, targets, format = 'plot'):
    '''

    :param logits:
    :return:
    '''
    nclasificadores, muestras, clases = logits.shape
    logits = np.argmax(logits, axis = 2, )
    if format == 'matrix':
        resultados = np.zeros((nclasificadores, nclasificadores))
    elif format == 'plot':
        resultados = []

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            comunes = np.equal(logits[i, :], logits[j, :])

            aciertos_comunes = np.sum(np.equal(logits[i, comunes], targets[comunes]))
            aciertos_totales = np.sum(np.equal(logits[i, :], targets))
            if aciertos_totales == 0:
                res = 0
            else:
                res = aciertos_comunes / aciertos_totales

            if format == 'matrix':
                resultados[i,j] = res
            elif format == 'plot':
                resultados.append((i, j, res))

    return resultados

def deep_belief_transform(X,  n_neurons = [70, 60, 50], n_layers = None,  batch_size = 100, num_epochs = 100):
    '''
    Fits a deep belief network to a X and returns both the transformed data and the fitted machine.
    '''
    import boltzmann_machine_deep as bmp
    
    activations = X > 0
    belief_network = bmp.DBN(n_neurons, n_layers)
    
    belief_network.fit(X, num_epochs, batch_size)
    
    return belief_network.transform(activations), belief_network
    
    
def pearson_distances(logits, format = 'plot'):
    '''

    :param logits:
    :return:
    '''
    from scipy.stats import pearsonr
    nclasificadores, muestras, clases = logits.shape
    if format == 'matrix':
        resultados = np.zeros((nclasificadores, nclasificadores))
    elif format == 'plot':
        resultados = []

    for i in range(nclasificadores):
        for j in range(nclasificadores):
            if format == 'matrix':
                resultados[i,j] = pearsonr(logits[i, :, 0], logits[j, :, 0])[0]
            elif format == 'plot':
                resultados.append((i,j, pearsonr(logits[i, :, 0], logits[j, :, 0])[0]))

    return resultados

def theoretical_limit(logits, labels):
    nclasificadores, muestras, clases = logits.shape
    preds = np.argmax(logits, axis = 2)
    aciertos = 0

    for y in range(len(labels)):
        acierto = np.max((labels[y] == preds[:, y]) > 0)
        aciertos += acierto

    return aciertos / muestras