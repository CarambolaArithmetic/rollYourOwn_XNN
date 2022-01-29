import numpy as np
from .math import *

#This is just a big wrapper of stuff to deal with the overall network
# includes an implementation of L2 Regularization, which is not being used.
class ErrorWithNormalizationTerms:
    #params:
    # netHead: the loss function or head of the network, such as softmax cross entropy.
    #      in other words, the entire network goes in this parameter.
    # lam: the regularization rate
    def __init__(self, netHead, lam):
        self.netHead = netHead
        self.lam = lam
        #this is where parameters we want to regularize will go
        self.paramsList = []

    #get error (regularization inclusive). should have been named loss()
    def error(self):
        return self.pureError() + self.normalization()

    #get purely the average predictive error
    def pureError(self):
        errors = self.netHead.fwd()
        n = errors.shape[-1]
        return (1/n)*sum(errors)

    #uses L2 Norm
    def normalization(self):
        return self.lam*np.sum([np.sum(W*W) for W in self.paramsList])

    #add another parameter to the regularization term
    def addNormalizationParams(self, W):
        self.paramsList.extend(W)

    #propogate the error and update all weights in the network
    #including L2 Norm if any params to norm are provided
    def propogateError(self, learningRate):
        errors = self.netHead.fwd()
        n = errors.shape[-1]
        [W.back(lambda: learningRate*self.lam*2*W.fwd()) for W in self.paramsList]
        self.netHead.back(lambda: learningRate/n)

    #get the network itself
    def getNetHead(self):
        return self.netHead
    #clear the network
    def clear(self):
        self.netHead.clear()


