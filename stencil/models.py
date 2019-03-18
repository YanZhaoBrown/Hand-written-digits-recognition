#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the Naive Bayes classifier

   Brown CS142, Spring 2019
"""
import random

import numpy as np


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        # TODO
        n = len(data.labels)
        self.P = []
        self.prob = np.zeros((self.n_classes, 784))
        
        for i in range(self.n_classes):
            index = np.where(data.labels == i) #Number of examples with label i
            self.P.append(len(index[0]) / n) #number of examples label i/N > total examples
            
        conditional = []
        for i in range(self.n_classes):
            tmp = data.inputs[data.labels==i]
            conditional.append(((tmp.sum(axis=0)+1)/(tmp.shape[0]+2)))
        self.prob = np.array(conditional)
        

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        #TODO
        product = np.copy(np.array(self.P))
        
        for i in range(self.n_classes):
            for j in range(784):
     
                product[i] = product[i] * self.prob[i][j]** inputs[j] * (1-self.prob[i][j])**(1 - inputs[j])
                
                      
        return np.argmax(product)
        

    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        #TODO
        n = len(data.labels)
        count = 0
        predict = np.zeros((n))
        
        for i in range(n):
            predict[i] = self.predict(data.inputs[i,:])
            if (predict[i] == data.labels[i]):
                count = count +1
          
        return (count/n)

