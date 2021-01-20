# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:50:16 2021

@author: javi-
"""
import torch

import numpy as np

from .. import bci_architectures as athena

from Fancy_aggregations import tensor_CCA as cca

supported_funcs = ['mean', 'max', 'sugeno', 'fsugeno', 'shamacher', 'choquet', 'cf', 'cf1f2']
# =============================================================================
# LABEL SMOOTH LOSS
# =============================================================================
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothCrossEntropyLoss(_WeightedLoss):
    '''
    Smooth labelling for pytorch.
    Source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    '''
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

# =============================================================================
#     DECISION LEARN FUNCTIONS FOR CCA
# =============================================================================
def unimodal_decision_learn(inputs, labels, agg1, agg2):
    '''
    Learns the best mixing parameter for a CCA using the two aggregations passed
    as parameters using the auto-diff from pytorch.

    :param inputs: training data: (frequencies, samples, clases).
    :param labels: labels (samples, )
    :param agg1: aggregation function. (Must have axis and keepdims params)
    :param agg2: aggregation function. (Must have axis and keepdims params)
    '''
    tensor_ag = cca.CCA_unimodal(agg1, agg2)

    criterion = SmoothCrossEntropyLoss()
    optimizer = torch.optim.SGD(tensor_ag.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    reshaped_inputs = np.swapaxes(inputs, 0, 1)
    trainloader = torch.utils.data.DataLoader(list(zip(reshaped_inputs, labels)), batch_size=4, shuffle=True)
    for epoch in range(10):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, labels = data
            labels = labels.type(torch.LongTensor)
            inputs1 = inputs1.permute(1, 0, 2)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = tensor_ag(inputs1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    return tensor_ag

def multimodal_decision_learn(inputs, labels, agg1, agg2, agg3, agg4):
    '''
    Learns the best mixing parameters for a CCA using the two aggregations passed
    as parameters in a multimodal decision using the auto-diff from pytorch.

    :param inputs: training data: (frequencies, samples, clases).
    :param labels: labels (samples, )
    :param agg1: aggregation function. (Must have axis and keepdims params)
    :param agg2: aggregation function. (Must have axis and keepdims params)
    :param agg3: aggregation function. (Must have axis and keepdims params)
    :param agg4: aggregation function. (Must have axis and keepdims params)
    '''
    numpy_version_inputs = np.zeros((len(inputs), inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]))
    for ix in range(len(inputs)):
        numpy_version_inputs[ix] = inputs[ix]

    tensor_ag = cca.CCA_multimodal((inputs[0].shape[0], 1, 1), agg1, agg2, agg3, agg4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(tensor_ag.parameters(), lr=0.001, momentum=0.9)
    reshaped_inputs = np.swapaxes(numpy_version_inputs, 0, 2)
    trainloader = torch.utils.data.DataLoader(list(zip(reshaped_inputs, labels)), batch_size=4, shuffle=True)
    for epoch in range(10):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, labels = data
            labels = labels.type(torch.LongTensor)
            inputs1 = inputs1.permute(2, 1, 0, 3)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = tensor_ag(inputs1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    return tensor_ag

# =============================================================================
# ARCHITECTURES
# =============================================================================
class bci_achitecture_cca(athena.bci_achitecture):
    '''
    Additional architectures for the sampling processing plugin. They implement
    the basic traditional and enhanced-traditional frameworks as described in the
    Transactions on Cybernetics paper. But they include support for the augmentation
    techniques.
    '''
    def trad_cca_architecture(self, X, y, verbose=False, agregacion=None):
        '''
        Traditional Fuzzy Framework using CCA aggregations.
        Check the docs for the EMF in the base module.
        
        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _predict_proba(Xf, csp_models, classifiers, ag1, num_classes):
                csp_x = athena.csp_forward(Xf, csp_models)
                logits = athena.fitted_trad_classifier_forward(csp_x, classifiers, self._num_classes)

                agg_output_tensor = ag1(torch.from_numpy(logits))

                return agg_output_tensor.detach().numpy()

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = athena.csp_train_layer(np.real(X), y)

        csp_og = csp_x
        y_og = y

        classifier = athena.trad_classifier_train(csp_x, y, designed_classifier=agregacion[1])

        logits = athena.fitted_trad_classifier_forward(csp_og, classifier, self._num_classes)
        ag1 = unimodal_decision_learn(logits, y_og, agregacion[0][0], agregacion[0][1])

        self._components = [csp_models, classifier, ag1, self._num_classes]
        self.predict_proba = lambda a: _predict_proba(a, csp_models, classifier, ag1, self._num_classes)

    def emf_cca_architecture(self, X, y, verbose=False, agregacion=None):
        '''
        Enhanced Multimodal Framework using CCA aggregations.
        Check the docs for the EMF in the base module.
        
        :param X: train (bands, time, samples)
        :param y: labels (samples,)
        '''
        def _emf_predict_proba(X, csp_models, classifiers, ag1, num_classes):
            csp_x = athena.csp_forward(X, csp_models)
            inputs = athena.fitted_classifiers_forward(csp_x, classifiers, self._num_classes)
            numpy_version_inputs = np.zeros((len(inputs), inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]))
            for ix in range(len(inputs)):
                numpy_version_inputs[ix] = inputs[ix]
            agg_output_tensor = ag1(torch.from_numpy(numpy_version_inputs))

            return agg_output_tensor.detach().numpy()

        self._num_classes = len(np.unique(y))

        csp_models, csp_x = athena.csp_train_layer(np.real(X), y)

        csp_og = csp_x
        y_og = y

        classifiers = athena.classifier_std_block_train(csp_x, y)

        logits = athena.fitted_classifiers_forward(csp_og, classifiers, self._num_classes)

        ag1 = multimodal_decision_learn(logits, y_og, agregacion[0][0], agregacion[0][1], agregacion[0][2], agregacion[0][3])

        self.predict_proba = lambda a: _emf_predict_proba(a, csp_models, classifiers, ag1, self._num_classes)
