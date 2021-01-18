#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:26:05 2020

Deep belief interface for the rest of the BCI project.

@author: fuminides
"""

from dbn.tensorflow import SupervisedDBNClassification, UnsupervisedDBN

# =============================================================================
# BOLTZMANN CLASS
# =============================================================================

class DBN:
    '''
    Deep belief neural network: it works using a tensorflow network underneath.
    I have made the API so that it is better at handling it from the rest of this library.
    
    Usage:
        1. Ceate object specifying size.
            If static: the number of neurons for each layer is specified.
            If dynamic: the number of layers and the first layer size is specified,
            we compute the number of neurons for the following layer using the
            entropy from a X tansformation of the actual machine.
        2. Fit the machine to the train data. 
        The train data must be samples x features. Num epochs and batch size should be specified.
        3. Transform subsequently X.
    '''
    def __init__(self, n_neurons = None, supervised=True):
        self.supervised = supervised
        if not n_neurons is None:
            self.create_classifier(n_neurons)

    def create_classifier(self, n_neurons = [10]):
        self.model = None
        self.n_neurons = n_neurons

    def fit(self, X, y=None, num_epochs=100, batch_size=32):
        '''
        Performs the X fit of a the Boltzmann machine object.
        '''
        if self.supervised:
            classifier = SupervisedDBNClassification(hidden_layers_structure=self.n_neurons,
                                 learning_rate_rbm=0.1,
                                 learning_rate=0.1,
                                 n_epochs_rbm=100,
                                 n_iter_backprop=200,
                                 batch_size=batch_size,
                                 activation_function='sigmoid',
                                 dropout_p=0.1)
            classifier.fit(X, y)
        else:                        
            classifier = UnsupervisedDBN(hidden_layers_structure=self.n_neurons,
                                                     learning_rate_rbm=0.1,
                                                     n_epochs_rbm=100,
                                                     batch_size=32,
                                                     activation_function='sigmoid')
            classifier.fit(X)         
        self.model = classifier

            
    def transform(self, X):       
        return self.model.transform(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def predict(self, X):
        if self.supervised:
            return self.model.predict(X)
        else:
            print('Something went wrong. You cant predict in non supervised mode!')
            
    def fit_transform(self, X, y, num_epochs=100, batch_size=100):
        self.fit(X, y, num_epochs=num_epochs, batch_size=batch_size)
        return self.transform(X)
    
