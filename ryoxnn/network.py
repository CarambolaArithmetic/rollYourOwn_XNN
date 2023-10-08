import numpy as np
from .math import *

#This is just a big wrapper of stuff to deal with the overall network.
class ErrorWithNormalizationTerms:
    #params:
    #   net_head: the loss function or head of the network, such as softmax cross entropy.
    #       in other words, the entire network goes in this parameter.
    #   lamb_da: The regularization rate. Also known as lambda, it is a numerical constant. not an anonymous function.
    def __init__(self, net_head, lamb_da):
        self.net_head = net_head
        self.lamb_da = lamb_da
        #this is where parameters we want to regularize will go
        self.params_list = []

    #Get value of the loss function (regularization inclusive).
    def loss(self):
        return self.pure_error() + self.l2_norm()

    #Get purely the average predictive error.
    def pure_error(self):
        errors = self.net_head.fwd()
        n = errors.shape[-1]
        return (1/n)*sum(errors)

    def l2_norm(self):
        return self.lamb_da*np.sum([np.sum(W*W) for W in self.params_list])

    #Add a weight array as an additional parameter to the regularization term.
    def add_normalization_params(self, weights):
        self.params_list.extend(weights)

    #Propogate the error and update all weights in the network,
    #   including L2 Norm if any params to norm are provided.
    def propogate_error(self, learning_rate):
        errors = self.net_head.fwd()
        n = errors.shape[-1]
        [W.bck(lambda: learning_rate*self.lamb_da*2*W.fwd()) for W in self.params_list]
        self.net_head.bck(lambda: learning_rate/n)

    #Get the network itself.
    def get_net_head(self):
        return self.net_head

    #Finalize the network.
    def finalize(self):
        self.net_head.finalize()


