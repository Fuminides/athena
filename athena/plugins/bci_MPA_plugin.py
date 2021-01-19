# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:15:16 2020

@author: JAVIER
"""
import numpy as np
import pandas as pd
import pickle

import bci_architectures as athena
import bci_penalty_plugin as bci_penalty
import load_brain_data as lb

from Fancy_aggregations import penalties as pn
from Fancy_aggregations import binary_parser as bp

quasi_similarities = [pn._cuadratic_cost, pn._realistic_optimistic_cost, pn._huber_cost, pn._realistic_pesimistic_cost]
quasi_dis = [pn._anti_cuadratic_cost]

quasi_total = quasi_similarities + quasi_dis
quasisim_name = ['Quadratic', 'Optimistic', 'Huber', 'Pessimistic']
quasidis_name = ['Anti-Quadratic']
names_total = quasisim_name + quasidis_name

def generate_combination(cost1, cost2):
    '''
    Returns a lambda of the two cost combined according to the appropiate mixing
    function.

    :param cost1: index of cost 1 in the quasi_total vector
    '''
    if (cost1 < len(quasi_similarities)) and (cost2 < len(quasi_similarities)):
        comb_function = pn._convex_comb
    else:
        comb_function = pn._convex_quasi_comb

    return comb_function(quasi_total[cost1], quasi_total[cost2])

# =============================================================================
# BINARY CLASSIFICATION GRADIENT FOR THE AGGREGATION MFF
# =============================================================================
def binary_cross_entropy_loss(output, y):
    return np.log(y * output + (1 - y) * output)

def binary_update_alpha(X, cost, alpha, y):
    forward_logits = multi_alpha_forward(X, cost, alpha)
    loss = binary_cross_entropy_loss(forward_logits[:,0], y)
    update_alpha = loss * X

    return update_alpha

def binary_train_loop(X, cost, alpha, y, learning_rate=0.01):
    '''
    '''
    loss = np.inf
    new_loss = 0

    while new_loss >= loss:
        loss = new_loss

        alpha += binary_update_alpha(X, cost, alpha, y)

        forward_logits = multi_alpha_forward(X, cost, alpha)
        new_loss = binary_cross_entropy_loss(forward_logits, y)

    return alpha, new_loss

multi_cost_names = ['R2+Opt', 'hub+Opt','Anti+Opt', 'Hub+Anti']
uni_cost_names = ['R2', 'opt', 'Anti-consensus', 'Huber', 'Pes']

# =============================================================================
# DYNAMIC ALPHA LEARNING METHOD
# =============================================================================
def logistic_alpha_forward(X, cost_convex, clf):
    '''
    X shape: (bandas, samples, clases)
    out shape: (samples, clases)
    '''

    reformed_X = np.swapaxes(X, 0, 1)
    reformed_X = reformed_X.reshape((reformed_X.shape[0], reformed_X.shape[1]*reformed_X.shape[2]))

    alphas = clf.predict(reformed_X)
    result = np.zeros((X.shape[1], X.shape[2]))

    for sample in range(X.shape[1]):
        alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alphas[sample])

        result[sample] = pn.penalty_aggregation(X[:, sample, :], [bp.parse(x) for x in athena.classical_aggregations], axis=0, keepdims=False, cost=alpha_cost)

    return result

def multimodal_alpha_forward(X, cost, cost2, alpha):
    '''
    X shape: list of n arrays (bandas, samples, clases)
    clfs: list of alphas.
    out shape: (samples, clases)
    '''
    david_played_and_it_pleased_the_lord = [bp.parse(x) for x in athena.agg_functions_names]
    agg_phase_1 = lambda X0, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X0, david_played_and_it_pleased_the_lord, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost(real, yhat, axis, alpha=alpha))
    agg_phase_2 = lambda X0, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X0, david_played_and_it_pleased_the_lord, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost2(real, yhat, axis, alpha=alpha))

    return mpa_aggregation(X, agg_phase_1, agg_phase_2, alpha, keepdims=False)


def generate_real_alpha(X, y, aggs, cost, opt=1):
    a = None
    b = None

    for alpha in np.arange(0.01, 1.01, 0.1):
        alpha_cost = lambda X, yhat, axis: cost(X, yhat, axis, alpha)
        pagg = pn.penalty_aggregation(X, [bp.parse(x) for x in aggs], axis=0, keepdims=False, cost=alpha_cost)

        if np.argmax(pagg) == y:
            if a is None:
                a = alpha
            else:
                b = alpha

    if a is None:
        a = 0.5
    if b is None:
        b = 0.5

    d1 = np.abs(a - 0.5)
    d2 = np.abs(b - 0.5)



    if opt == 1:
        if d1 <= d2:
            return a
        else:
            return b

    elif opt == 2:
        return (a + b) / 2

def generate_train_data_alpha(logits, labels, aggs=athena.classical_aggregations, cost=pn.cost_functions[0], opt=1):
    bands, samples, classes = logits.shape

    y = np.zeros((samples,))

    for sample in range(samples):
        y[sample] = generate_real_alpha(logits[:,sample,:], labels[sample], aggs, cost, opt=opt)

    return y


def gen_all_good_alpha_trad(cost, aggs=athena.classical_aggregations, opt=1, multi_class=False):
    '''
    Learn the logistic regression for the whole set of datasets. (Trad framework)
    '''
    import sklearn.linear_model
    from sklearn.model_selection import KFold

    n_splits = 5
    kf = KFold(n_splits=n_splits)
    all_data = carmen_penalty_cache_logits(mff=False, multi_class=multi_class)
    all_logits_train, all_y_train, all_logits_test, all_y_test = all_data
    clfs = []


    for logits_train, y_train, logits_test, y_test in zip(all_logits_train, all_y_train, all_logits_test, all_y_test):
        if opt == 1 or opt == 2:
            logits_reshaped = np.swapaxes(logits_train, 0, 1)
            logits_reshaped = logits_reshaped.reshape((logits_reshaped.shape[0], logits_reshaped.shape[1]*logits_reshaped.shape[2]))
            y_alpha = generate_train_data_alpha(logits_train, y_train, aggs=aggs, cost=cost, opt=opt)
            clf = sklearn.linear_model.LinearRegression().fit(logits_reshaped, y_alpha)
        elif opt == 'classic':
            def func_opt(alphita):
                acc = lambda x_data, y : np.mean(np.equal(y, np.argmax( pn.penalty_aggregation(x_data, [bp.parse(x) for x in aggs], axis=0, keepdims=False, cost=lambda X, yhat, axis: cost(X, yhat, axis, alpha=alphita)), axis=1)))
                acum_acc = 0
                for train_index, test_index in kf.split(logits_train[0,:,:]):
                    _, X_test = logits_train[:, train_index, :], logits_train[:, test_index, :]
                    _, y_test = y_train[train_index], y_train[test_index]
                    acum_acc += acc(X_test, y_test)

                return 1 - acum_acc / n_splits #Rememeber: we are minimizing

            clf = athena.my_optimization(func_opt)

        clfs.append(clf)

    return clfs
# =============================================================================
# MULTIMODAL ALPHA OPTIMIZATION
# =============================================================================
def eval_alpha(alpha_v, y_hat, y):
    '''


    Returns
    -------
    None.

    '''
    alpha_score = np.mean(np.minimum(alpha_v, 1 - alpha_v))
    acc_score = np.mean(np.equal(y_hat, y))

    return (alpha_score + acc_score) / 2

def mpa_aggregation(logits, agg1, agg2, alpha, keepdims=False):
    n_2 = len(logits)
    n_1, samples, clases = logits[0].shape

    res = np.zeros((n_2, samples, clases))

    for ix, logit in enumerate(logits):
        res[ix, :, :] = agg1(logit, axis=0, keepdims=False, alpha=alpha[ix])

    return agg2(res, axis=0, keepdims=keepdims, alpha=alpha[-1])

def eval_conf(X, alpha, y, agg1, agg2):
    '''
    Computes the mpa agg for X, and returns the optimization score.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    agg1 : TYPE
        DESCRIPTION.
    agg2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    y_hat = np.argmax(mpa_aggregation(X, agg1, agg2, alpha), axis=1)
    return eval_alpha(alpha, y_hat, y)

def gen_all_good_alpha_mff(cost, cost2, aggs=athena.agg_functions_names, opt=1, four_class=False):
    '''
    Learn the logistic regression for the whole set of datasets.
    '''
    from scipy.optimize import least_squares
    all_data = carmen_penalty_cache_logits(mff=True, multi_class=four_class)
    all_logits_train, all_y_train, all_logits_test, all_y_test = all_data

    datasets = []
    agg_phase_1 = lambda X, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X, aggs, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost(real, yhat, axis, alpha=alpha))
    agg_phase_2 = lambda X, alpha, keepdims=False, axis=0: pn.penalty_aggregation(X, aggs, axis=axis, keepdims=keepdims, cost=lambda real, yhat, axis: cost2(real, yhat, axis, alpha=alpha))

    for logits_train, y_train, logits_test, y_test in zip(all_logits_train, all_y_train, all_logits_test, all_y_test):
        optimize_lambda = lambda alpha: -eval_conf(logits_train, alpha, y_train, agg_phase_1, agg_phase_2) #Remember we are minimizng
        x0_alpha = np.array([0.5] * len(logits_train) + [0.5])

        res_1 = least_squares(optimize_lambda, x0_alpha, bounds=[0.0001, 0.9999])
        datasets.append(res_1.x)


    return datasets

# =============================================================================
# SAVE AND LOAD CLFS
# =============================================================================
def save_clfs(cost, cost_name, mff=False, four_classes=False, opt=1):
    if mff:
        clfs = gen_all_good_alpha_mff(cost, cost, [bp.parse(x) for x in athena.mff_aggregations], opt=opt, four_class=four_classes)
    else:
        clfs = gen_all_good_alpha_trad(cost, athena.classical_aggregations, opt=opt, multi_class=four_classes)

    ending = ''
    if not mff:
        ending += '_trad'
    else:
        ending += '_mff'

    if four_classes:
        ending += '_tonguefoot'
    else:
        ending += '_binary'

    ending += '_' + str(opt)

    if (opt == 1) or (opt == 2):
        with open('./regression_models/lclfs_' + cost_name + ending + '.pckl', 'wb') as f:
            pickle.dump(clfs, f)
    elif opt == 'classic':
        with open('./classic_alpha_models/alpha_' + cost_name + ending + '.pckl', 'wb') as f:
            pickle.dump(clfs, f)

def load_clfs(cost_name, mff=False, four_classes=False, opt=1):
    import pickle
    ending = ''
    if not mff:
        ending += '_trad'
    else:
        ending += '_mff'

    if four_classes or four_classes == '_4_class':
        ending += '_tonguefoot'
    else:
        ending += '_binary'

    ending += '_' + str(opt)

    if (opt == 1) or (opt == 2):
        with open('./regression_models/lclfs_' + cost_name + ending + '.pckl', 'rb') as f:
            clfs = pickle.load(f)
    elif opt == 'classic':
        with open('./classic_alpha_models/alpha_' + cost_name + ending + '.pckl', 'rb') as f:
            clfs = pickle.load(f)

    return clfs

def compute_accuracy_logistic_alpha(cost_convex, cost_name, mff=True, four_classes=False, opt=1):
    accuracy = lambda yhat, yreal: np.mean(np.equal(yhat, yreal))
    clfs = load_clfs(cost_name, mff, four_classes, opt)

    all_logits_train, all_y_train, all_logits_test, all_y_test = carmen_penalty_cache_logits(mff=mff, multi_class=four_classes)
    train_accuracies = np.zeros((len(all_y_train)))
    test_accuracies = np.zeros((len(all_y_test)))
    ix = 0

    for logits_train, y_train, logits_test, y_test in zip(all_logits_train, all_y_train, all_logits_test, all_y_test):
        if mff:
            agg_logits = multimodal_alpha_forward(logits_train, cost_convex, cost_convex, clfs[ix])
            agg_logits_test = multimodal_alpha_forward(logits_test, cost_convex, cost_convex, clfs[ix])
        else:
            if (opt == 1) or (opt == 2):
                agg_logits = logistic_alpha_forward(logits_train, cost_convex, clfs[ix])
                agg_logits_test = logistic_alpha_forward(logits_test, cost_convex, clfs[ix])
            elif opt == 'classic':
                alpha_cost = lambda X, yhat, axis: cost_convex(X, yhat, axis, alpha=clfs[ix])
                agg_logits = bci_penalty._forward_penalty(logits_train, alpha_cost)
                agg_logits_test = bci_penalty._forward_penalty(logits_test, alpha_cost)

        yhat_train = np.argmax(agg_logits, axis=1)
        yhat_test = np.argmax(agg_logits_test, axis=1)
        train_accuracies[ix] = accuracy(yhat_train, y_train)
        test_accuracies[ix] = accuracy(yhat_test, y_test)

        ix+= 1

    return train_accuracies, test_accuracies

# =============================================================================
# FORWARD AND MISCELLANIOUS
# =============================================================================
def carmen_penalty_cache_logits(mff=True, multi_class=False):
        if mff:
            framework='mff'
            archi = athena.bci_achitecture.emf_carmen_penalty_architecture
        else:
            framework='trad'
            archi = athena.bci_achitecture.trad_carmen_penalty_architecture

        if multi_class or (multi_class == '_4_class'):
            class_mode = '4_class'
        elif not multi_class or (multi_class == '_binary'):
            class_mode = 'binary'

        try:
            with open('logits_' + framework + '_carmen_' + class_mode + '_train.pckl', 'rb') as f:
                 logits_train = pickle.load(f)
            with open('logits_' + framework + '_carmen_'+ class_mode +'_test.pckl', 'rb') as f:
                 logits_test = pickle.load(f)
            with open('logits_' + framework + '_carmen_'+ class_mode +'_train_y.pckl', 'rb') as f:
                 y_train = pickle.load(f)
            with open('logits_' + framework + '_carmen_'+ class_mode +'_test_y.pckl', 'rb') as f:
                 y_test = pickle.load(f)
        except (IOError, EOFError) as e:
            print('Recomputing logits...', e)
            logits_train, y_train, logits_test, y_test = carmen_best_alpha_2a_all_logits(archi, 0, verbose=False, agregacion=pn.base_cost_functions[0], mff=mff, four_clases=multi_class)

        return logits_train, y_train, logits_test, y_test



def multi_alpha_forward(X, cost_convex, alpha):
    try:
        multiple_alpha = len(alpha) > 1
    except:
        multiple_alpha = False

    agg_logits = np.zeros((len(X), X[0].shape[1], X[0].shape[2]))
    for wave in range(len(X)):
        if not multiple_alpha:
            alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha[0])
        else:
            alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha[wave])

        agg_logits[wave] = pn.penalty_aggregation(X[wave], [bp.parse(x) for x in athena.mff_aggregations], axis=0, keepdims=False, cost=alpha_cost)

    if multiple_alpha:
        alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha[-1])
    else:
        alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha)

    return pn.penalty_aggregation(agg_logits, [bp.parse(x) for x in athena.mff_aggregations], axis=0, keepdims=False, cost=alpha_cost)

def _alpha_learn(X, y, cost, mff=False):

    def compute_accuracy(yhat, y):
        return np.mean(np.equal(yhat, y))

    def optimize_function(X, y, cost_convex, alpha):
        alpha_cost = lambda real, yhat, axis: cost_convex(real, yhat, axis, alpha)

        agg_logits = pn.penalty_aggregation(X, [bp.parse(x) for x in athena.classical_aggregations], axis=0, keepdims=False, cost=alpha_cost)
        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    def optimize_function_mff(X, y, cost_convex, alpha):
        agg_logits = multi_alpha_forward(X, cost_convex, alpha)
        yhat = np.argmax(agg_logits, axis=1)

        return 1 - compute_accuracy(yhat, y)

    if mff:
        function_alpha = lambda a: optimize_function_mff(X, y, cost, a)
        x0 = [0.5] * 5
    else:
        function_alpha = lambda a: optimize_function(X, y, cost, a)
        x0 = [0.5]


    res = athena.my_optimization(function_alpha, x0=x0, niter=100, mode='montecarlo')
    alpha_value = res

    if hasattr(alpha_value, 'len'):
        alpha_value = alpha_value[0]

    return lambda real, yhatf, axis: cost(real, yhatf, axis, alpha_value), alpha_value

def study_alpha(architecture, x_test, y_test, cost, mff=False):
    lgts = architecture.csp_pass(x_test)
    alpha_f, alpha = _alpha_learn(lgts, y_test, cost, mff=mff)
    aggregation_set = athena.classical_aggregations if not mff else athena.mff_aggregations

    if not mff:
        agg_logits = pn.penalty_aggregation(lgts, [bp.parse(x) for x in aggregation_set], axis=0, keepdims=False, cost=alpha_f)
    else:
        agg_logits = multi_alpha_forward(lgts, cost, alpha)

    yhat = np.argmax(agg_logits, axis=1)
    return np.mean(np.equal(y_test, yhat))

def carmen_train_2a_all(architecture, derivate=0, verbose=False, agregacion=None, four_clases=False, opt=False):
    accuracies = []

    for dataset_train, dataset_test in zip(lb.all_carmen_datasets_partitions(full_tasks=four_clases, opt=opt), lb.all_carmen_datasets_partitions(full_tasks=four_clases, test=True, opt=opt)):
        X_train, y_train = dataset_train
        X_test, y_test = dataset_test

        X_train = np.transpose(X_train, (2, 1, 0))
        X_test = np.transpose(X_test, (2, 1, 0))

        my_architecture = athena.bci_achitecture()
        if agregacion is None:
            architecture(my_architecture, X_train, y_train, verbose)
        else:
            architecture(my_architecture, X_train, y_train, verbose, agregacion)

        accuracies.append(np.mean(np.equal(my_architecture.forward(X_test), y_test)))

    if verbose:
        print('N-th accuracy: ' + str(np.mean(np.equal(my_architecture.forward(X_test), y_test))), 'Actual mean: ' + str(np.mean(accuracies)))

    return accuracies, my_architecture

def carmen_best_alpha_2a_all(architecture, cost, derivate=0, verbose=False, agregacion=None, four_clases=False, opt=False, mff=False):
    accuracies = []

    for dataset_train, dataset_test in zip(lb.all_carmen_datasets_partitions(full_tasks=four_clases, opt=opt), lb.all_carmen_datasets_partitions(full_tasks=four_clases, test=True, opt=opt)):
        X_train, y_train = dataset_train
        X_test, y_test = dataset_test

        X_train = np.transpose(X_train, (2, 1, 0))
        X_test = np.transpose(X_test, (2, 1, 0))

        my_architecture = athena.bci_achitecture()
        if agregacion is None:
            architecture(my_architecture, X_train, y_train, verbose)
        else:
            architecture(my_architecture, X_train, y_train, verbose, agregacion)

        accuracies.append(study_alpha(my_architecture, X_test, y_test, agregacion, mff))

    if verbose:
        print('N-th accuracy: ' + str(np.mean(np.equal(my_architecture.forward(X_test), y_test))), 'Actual mean: ' + str(np.mean(accuracies)))

    return accuracies, my_architecture

def carmen_best_alpha_2a_all_logits(architecture, cost, derivate=0, verbose=False, agregacion=None, four_clases=False, opt=False, mff=False):
    logits_train = []
    logits_test = []
    all_y_train = []
    all_y_test = []
    n_partitions = 20

    for dataset_train, dataset_test in zip(lb.all_carmen_datasets_partitions(full_tasks=four_clases, opt=opt, n_partition=n_partitions), lb.all_carmen_datasets_partitions(full_tasks=four_clases, test=True, opt=opt, n_partition=n_partitions)):
        X_train, y_train = dataset_train
        X_test, y_test = dataset_test

        X_train = np.transpose(X_train, (2, 1, 0))
        X_test = np.transpose(X_test, (2, 1, 0))

        my_architecture = athena.bci_achitecture()
        if agregacion is None:
            architecture(my_architecture, X_train, y_train, verbose)
        else:
            architecture(my_architecture, X_train, y_train, verbose, agregacion)

        logits_train.append(my_architecture.logits)
        logits_test.append(my_architecture.csp_pass(X_test))
        all_y_train.append(y_train)
        all_y_test.append(y_test)

    import pickle
    framework = 'mff' if mff else 'trad'
    class_mode = 'binary' if not four_clases else '4_class'

    with open('logits_' + framework + '_carmen_' + class_mode + '_train.pckl', 'wb') as f:
        pickle.dump(logits_train, f)
    with open('logits_' + framework + '_carmen_' + class_mode + '_train_y.pckl', 'wb') as f:
        pickle.dump(all_y_train, f)
    with open('logits_' + framework + '_carmen_' + class_mode + '_test.pckl', 'wb') as f:
        pickle.dump(logits_test, f)
    with open('logits_' + framework + '_carmen_' + class_mode + '_test_y.pckl', 'wb') as f:
        pickle.dump(all_y_test, f)

    return logits_train, all_y_train, logits_test, all_y_test

def classifier_new_alpha(nombre, coste, multi_class=False, reload=True, mff=False, opt=1):
    if reload:
        save_clfs(coste, nombre, mff, four_classes=multi_class, opt=opt)

    train_acc, test_acc = compute_accuracy_logistic_alpha(coste, nombre, mff, multi_class, opt)

    return train_acc, test_acc

def accuracy_report(cost_names, cost_funcs, mff, reload, multi_class, opt):
    accuracies_df = pd.DataFrame(np.zeros((len(cost_names), 1)), index=cost_names, columns=['Accuracy'])
    for ix, cost_func in enumerate(cost_funcs):
        accuracies_train, accuracies_test = classifier_new_alpha(cost_names[ix], cost_func, mff=mff, reload=reload, multi_class=multi_class, opt=opt)
        accuracies_df.iloc[ix,0] = np.mean(accuracies_test)
    return accuracies_df

def accuracy_report_dual(mff, reload, multi_class_bool, opt):
    '''
    Igual que accuracy report pero en una table de coste x coste
    '''
    accuracies_df = pd.DataFrame(np.zeros((len(names_total), len(names_total))), index=names_total, columns=names_total)
    for ix, cost_func in enumerate(quasi_total):
        for jx, cost_func2 in enumerate(quasi_total):
            cost = generate_combination(ix, jx)
            accuracies_train, accuracies_test = classifier_new_alpha(names_total[ix] + '+' + names_total[jx], cost, mff=mff, reload=reload, multi_class=multi_class_bool, opt=opt)
            acc_test_df = pd.DataFrame(accuracies_test)
            mff_str = 'mff' if mff else 'trad'
            multi_class = 'four_classes' if multi_class_bool else 'binary'
            acc_test_df.to_csv('./carmen_results/' + mff_str + '_' + multi_class + '_' + str(opt) + '_' + names_total[ix].replace(' ','_') + '_' + names_total[jx].replace(' ','_') + '.csv')
            accuracies_df.iloc[ix,jx] = np.mean(accuracies_test)

    return accuracies_df

def bin_alpha(opt=1, cost_names=multi_cost_names, cost_funcs=(pn.cost_functions[0], pn.cost_functions[1], pn.cost_functions[2], pn.cost_functions[3]), multi_class=False, mff=False, reload=True):
        framework = 'mff' if mff else 'trad'
        opt_str = '_' + str(opt)
        multiclass_str = '_4_class' if multi_class else '_binary'
        accuracy_report(cost_names, cost_funcs, mff, reload, multi_class, opt).to_csv('carmen_' + framework + '_penalty_costs_accuracies_best_alpha' + opt_str + multiclass_str + '.csv')

def bin_alpha_dual(opt=1, multi_class=False, mff=False, reload=True):
    framework = 'mff' if mff else 'trad'
    opt_str = '_' + str(opt)
    multiclass_str = '_4_class' if multi_class else '_binary'
    accuracy_report_dual(mff, reload, multi_class, opt).to_csv('carmen_' + framework + '_penalty_costs_accuracies_best_alpha' + opt_str + multiclass_str + '.csv')

def convert_to_tex(path_df):
    '''
    '''
    load_df = pd.read_csv(path_df, index_col=0)
    load_df.values[[np.arange(load_df.shape[0])]*2] = 0
    max_value = load_df.max().max()
    n_cols = len(load_df.columns)

    def normal_format(val):
        return val

    def max_bold(val):
        max_string = '$ %.4f $' if val != max_value else '{$ \\mathbf{%.4f} $}'

        return max_string % val

    load_df.to_latex(open(path_df.replace('.csv', '.tex'), 'w'), formatters=[max_bold]*n_cols, column_format='l' + 'c'*n_cols, escape=False)

def carmen_main(mode, args=None):
    print('Carmen bci plugin, mode: ', mode)
    # =============================================================================
    #                 CARMEN CSP PENALTY DATA EXPERIMENTS
    # =============================================================================
    if str(mode) == 'study_new_alpha_costs_binary':
        bin_alpha_dual(opt='classic', multi_class=False, mff=False, reload=True)
        bin_alpha_dual(opt=1, multi_class=False, mff=False, reload=True)

    elif str(mode) == 'study_new_alpha_costs_full_class':
        bin_alpha_dual(opt='classic', multi_class=True, mff=False, reload=True)
        bin_alpha_dual(opt=1, multi_class=True, mff=False, reload=True)

    elif str(mode) == 'study_new_alpha_costs_binary_mff':
        bin_alpha_dual(opt=1, multi_class=False, mff=True, reload=True)

    elif str(mode) == 'study_new_alpha_costs_full_class_mff':
        bin_alpha_dual(opt=1, multi_class=True, mff=True, reload=True)

    else:
        return 0

    return 1

if __name__ == '__main__':
    carmen_main('study_new_alpha_costs_binary_mff')