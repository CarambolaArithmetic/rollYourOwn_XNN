import numpy as np
from .math import *


class ErrorWithNormalizationTerms:
    """
    Wrapper for a network with normalization terms.
    """

    def __init__(self, net_head, lamb_da):
        """
        Parameters
        ----------
            net_head: 
                the loss function or head of the network, such as softmax cross entropy.
                in other words, the entire network goes in this parameter.
            lamb_da: 
                The normalization rate. Also known as lambda, it is a numerical constant. not an anonymous function.
        """
        self.net_head = net_head
        self.lamb_da = lamb_da
        # this is where parameters we want to normalize will go
        self.params_list = []

    def loss(self):
        """
        Get value of the loss function (normalization inclusive).
        """
        return self.pure_error() + self.l2_norm()

    def pure_error(self):
        """
        Get purely the average predictive error.
        """
        errors = self.net_head.fwd()
        n = errors.shape[-1]
        return (1/n)*sum(errors)

    def l2_norm(self):
        return self.lamb_da*np.sum([np.sum(W*W) for W in self.params_list])

    def add_normalization_params(self, weights):
        """
        Add a weight array as an additional parameter to the normalization term.
        """
        self.params_list.extend(weights)

    def propogate_error(self, learning_rate):
        """
        Propogate the error to all weights in the network,
        including L2 Norm if any params to norm are provided.
        """

        errors = self.net_head.fwd()
        n = errors.shape[-1]
        [W.bck(lambda: learning_rate*self.lamb_da*2*W.fwd())
         for W in self.params_list]
        self.net_head.bck(lambda: learning_rate/n)

    def get_net_head(self):
        """
        Get the network itself.
        """
        return self.net_head

    def finalize(self):
        """
            Apply weight update calculated from the last call to propagate_error.
        """
        self.net_head.finalize()
