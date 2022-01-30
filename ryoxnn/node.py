try:
        import cupy as np
except ImportError:
        import numpy as np
from .math import  *

#Graph Node. Caches the value of the calculation made in the forward pass in self.value
#   all operations inherit from this. it provides the public interface methods, fwd, and back,
#   which encapsulate the class-by-class implementation of those operations.
#   this doesn't actually do anything really, it's mostly just an aid to ensure that my classes
#   follow the correct structure and also manage cached operations so that they aren't calculated every single time
#   a part of the network needs a forward pass from one of its children.
class Node:
    #parameters:
    #params: a list of parameter names that might be cleared from memory when clear() is called
    # needsUpdate: whether or not this node (and thus any of its children) needs errors propogated to it
    # in the backward pass. this would be false, for instance, if all of this node's children were input tensors
    # (as in, the images in the input of the network
    def __init__(self, params, needsUpdate = None):
        self.value = None
        self.set = False
        self.params = params
        #calculate the grad if any of its children need it.
        if needsUpdate != None:
            self.needsUpdate = needsUpdate
        else:
            self.needsUpdate = any([self.__dict__[p].getNeedsUpdate() for p in self.params])

    #after the first call to fwd(), any subsequent call to fwd() will actually return a cached value.
    #calling this clear() method tells the Node to do a fresh recalculation the next time fwd() is called
    def clear(self):
        self.value = None
        self.set = False
        [self.__dict__[p].clear() for p in self.params]
    #get the forward pass for this node.
    #if this node has already been calculated, return that value. otherwise, calculate that value and return it
    def fwd(self):
        if self.set == False:
            self.value = self._forward()
            self.set = True
        return self.value
    #propogate the error to this node.
    #param: getDeL
    #   this should be a lambda that calculates the gradient from the parent, with the intent that that calculation
    #   does not need to occur unless this node actually needs it.
    def back(self, getDeL):
        #only compute the gradient if an update is actually needed
        if self.needsUpdate:
            deL = getDeL()
            self._back(deL)
    # return true if this node or any of its children have parameters that backprop needs to update
    def getNeedsUpdate(self):
        return self.needsUpdate
    #business logic for forward pass
    def _forward(self):
        raise Exception("forward unimplemented!")
    #business logic for backward pass
    #param: deL
    #   gradient from parent (np.ndarray, already calculated
    def _back(self, deL):
        raise Exception("backward unimplemented!")

#Input tensor.
#is not really a node, does not result from an operation, but all node classes need to treat it that way
#because they need to assume all of their children function as nodes
#stores a numpy array as its "value", and has an update rule which determines how it will update that value
#when given a gradient in the back pass.
class Tensor:
    def __init__(self, A = np.empty([]), updateRule = lambda x, y: x - y):
        self.value = A
        self.updateRule = updateRule

    #this was an attempt at a performance increase to reduce unneeded operations,
    #unfortunately it doesn't actually make that much sense; kept it in due to time constraints
    def getNeedsUpdate(self):
        return self.updateRule != None #needs update if we have updateRule

    #just return value
    def fwd(self):
        if self.value.size == 0:
            raise Exception("tensor is unset!")
        return self.value

    #notice that this will likely be called many times during a backward pass
    # it is the sum of these deLs that provides the true gradient, not each
    # individual one.
    def back(self, getDeL):
        if self.value.size == 0:
            raise Exception("tensor is unset!")
        if self.updateRule != None:
            deLVal = getDeL()
            self.value = self.updateRule(self.value, deLVal)

    #Tensors often need to have their internal values changed after initialization.
    #example: inputs to the networks, labels.
    def set(self,A):
        self.value = A
    def clear(self):
        #the value we hold is not the cached result of an operation
        #so don't clear it when asked
        return

#Node that implements basic matrix multiplication
class Matmul(Node):
    #params:
    #W is weights , of size w x m
    #X is input, of size m x n where n is the number of inputs in the batch.
    def __init__(self, W,X):
        self.W = W
        self.X = X
        super().__init__(["W","X"])
    def _forward(self):
        return np.matmul(self.W.fwd(),self.X.fwd())
    def _back(self, deL):
        self.W.back(lambda: np.matmul(deL,np.transpose(self.X.fwd())))
        #this could be more efficient, but I know it works.
        self.X.back(lambda: np.transpose(np.matmul(np.transpose(deL),self.W.fwd())))

#Node that implements Softmax with logistic regression
class SoftmaxWithLogit(Node):
    #O for output (of previous layer and D for desired response
    #D for Desired Labels
    #D isn't updated. I don't know what the derivative relative to it is, nor do I care.
    #O is m x n
    #D is m x n
    def __init__(self, D,O):
        self.O = O
        self.D = D
        self.softmax = None
        super().__init__(["D","O"])
    def _forward(self):
        self.softmax = np.exp(self.O.fwd())/np.sum(np.exp(self.O.fwd()),axis=0)
        return -np.sum(self.D.fwd()*np.log(self.softmax), axis = 0)
    def _back(self, deL):
        #only propogate to Y, because that's coming in from elsewhere in the net.
        self.O.back(lambda: deL*(self.softmax-self.D.fwd()))
    #get the output of the softmax.
    def getSoftmax(self):
        return self.softmax
    #get 1 x n vector containing predicted labels of network input
    def getPredictions(self):
        return np.argmax(self.softmax, axis = 0)

class ReLu(Node):
    def __init__(self, A):
        self.A = A
        super().__init__(["A"])
    def _forward(self):
        matt = self.A.fwd()
        return np.max(np.array([matt, np.zeros(matt.shape)]), axis=0)
    def _back(self, deL):
        self.A.back(lambda: deL * np.where(self.value > 0, 1, 0))

#node that implements a leaky relu. This was used instead of regular relu because it was found to be more stable
class LeakyReLu(Node):
    #params:
    #   A: m x n matrix
    def __init__(self, A):
        self.A = A
        super().__init__(["A"])
    def _forward(self):
        matt = self.A.fwd()
        return np.max(np.array([matt, 0.1*np.ones(matt.shape)]), axis=0)
    def _back(self, deL):
        self.A.back(lambda: deL * np.where(self.value > 0, 1, 0.1))

#Node that implementes a *concatenation# between two inputs A and B, along axis.
#this is useful for matmulWBias
class Cat(Node):
    #A is m x n, B is K x N, returns a (m+k) x n matrix
    def __init__(self, A, B, axis):
        self.A = A
        self.B = B
        self.axis = axis
        self.splitMarker = None
        super().__init__(["A","B"])
    def _forward(self):
        self.splitMarker = self.A.fwd().shape[self.axis]
        return np.concatenate([self.A.fwd(), self.B.fwd()], axis=self.axis)
    #back pass just splits gradient into two parts, each one of which corresponds with A or B
    def _back(self, deL):
        [deLA, deLB] = np.split(deL, [self.splitMarker], axis=self.axis)
        self.A.back(lambda: deLA)
        self.B.back(lambda: deLB)

#Node that implements a matrix multiplication with a bias term
#this is the same thing as doing matmul and then an addition,
# it just rolls the bias into the matrix operation by appending a 1 x n vector of ones along the input's m dimension
# essentially this means we're just storing our addition weights on the end of the matmul weight matrix
class MatmulWBias(Node):
    #params: A is weights matrix of size w x m+1
    #B is input of size m x n
    def __init__(self, A, B, axis=0):
        self.A = A
        self.B = B
        self.C = None
        self.axis = axis
        #C is a result of operations on A and B, so no need to clear those
        #they get cleared when C gets cleared
        super().__init__(["C"], True)
    def _forward(self):
        bShape = list(self.B.fwd().shape)
        bShape[self.axis] = 1
        oneVec = Tensor(np.ones(bShape),None)
        #this operation can be defined as a composition of other ones we've already defined! excellent!
        self.C = Matmul(self.A, Cat(self.B, oneVec, self.axis))
        return self.C.fwd()
    def _back(self, deL):
        self.C.back(lambda: deL)

#note, only works with square inputs, as far as I know
#dims explanation:
#f: number of filters (output channels)
#c: number of channels (input channels)
#n: number of images
#l: output height and width
#tensor operation output shape is l, l, f, n
#designed to work in both directions(!) with strided convolution, but
#the strided part was never used
class Convolve(Node):
    def __init__(self, X, K, stride, l):
        #kernel, shape: m,m,c,f
        self.K = K

        #input feature map, shape: x, x, c, n
        self.X = X

        #desired output size, needed to determine padding
        self.l = l
        self.stride = stride
        super().__init__(["K","X"])
    def _forward(self):
        x0 = self.X.fwd().shape[0]
        k0 = self.K.fwd().shape[0]
        self.xPaddingSize = paddingSize(k0, x0, self.stride, self.l)
        self.paddedInput = padImage(self.X.fwd(), self.xPaddingSize)
        return convoolv3(self.paddedInput, self.K.fwd(), self.stride)

    #source for the backprop rules i'm using:
    #   https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
    #Works with strided convolution, probably.
    def _back(self,deL):
        def XBack():
            deLDialed = dialate(deL, self.stride-1)
            deLPad = padImage(deLDialed, paddingSize(self.K.fwd().shape[0],
                                                     deLDialed.shape[0],
                                                     self.stride, self.paddedInput.shape[0]))
            flippedK = np.flip(np.flip(self.K.fwd(), axis=0), axis=1)
            deLToXPad = backConvoolvX(deLPad, flippedK,1)
            #deLToXPad is the gradient with respect to the PADDED version of X, so we need to clip off the parts
            #that are with respect to the zero-padding.
            return getClipped(deLToXPad,self.xPaddingSize)
        def KBack():
            return backConvoolvK(self.paddedInput, dialate(deL, self.stride-1),1)
        self.K.back(KBack)
        self.X.back(XBack)

#implements a max pool of a 4th order tensor
class SquareMaxPool(Node):
    def __init__(self,X,windowSize, stride):
        self.X = X
        self.windowSize = windowSize
        self.stride = stride
        super().__init__(["X"])

    #this is incredibly ugly but I don't care anymore
    def _forward(self):
        x0,x1,c,n = self.X.fwd().shape

        #get the max of the window over X whose top corner is m0, m1
        #sets the max coordinate and returns the max value with coordinates in term of X.
        def getMax(m0,m1):
            m02 = (m0+ self.windowSize)
            m12 = (m1 +self.windowSize)
            field = self.X.fwd()[m0:m02,m1:m12,:,:]
            spots = np.array([[np.unravel_index(field[:, :, c, n].argmax(), field[:, :, c, n].shape) for c in range(0, field.shape[2])]
                for n in range(0, field.shape[3])])
            #apply offset
            #dims: n x c x pt
            #where [n,c,:] is the coordinate tuple x0 x1
            spots[:,:,0] = spots[:,:,0] + m0
            spots[:,:,1] = spots[:,:,1] + m1
            # c x n
            maxes = np.max(np.max(field, axis=0), axis=0)
            return (maxes,np.transpose(spots))
        self.numPools = maxPoolOutSize(self.windowSize, self.X.fwd().shape[0], self.stride)
        self.backWindows = np.zeros([self.numPools,self.numPools, 2, c, n])
        self.output = np.zeros([self.numPools,self.numPools, c, n])
        for i in range(0,self.numPools):
            for k in range(0,self.numPools):
                maxVals, argMaxes = getMax(i*self.stride, k*self.stride)
                self.backWindows[i,k,:,:,:] = argMaxes
                self.output[i,k,:,:] = maxVals
        return self.output
    def _back(self, deL):
        def bacc():
            newDeL = np.zeros(self.X.fwd().shape)
            x0, x1, xc, xn = self.X.fwd().shape
            #i'm done. i don't have an elegant solution to this
            for i in range(0,self.numPools):
                for k in range(0, self.numPools):
                    for c in range(0,xc):
                        for n in range(0,xn):
                            #we get the pair p0 p1 stored in the backWindows matrix that
                            #corresponds with the output feature at i,k,c,n.,
                            #these are the h, w coords in the input for the max that was used in that output
                            #we then add deL[i,k,c,n] to the our new gradient tensor (newDeL) at that coord.
                            #so if two windows have the same max, two different gradients get added to that point
                            #in the backward tensor.
                            pt = self.backWindows[i,k,:,c,n]
                            newDeL[int(pt[0]),int(pt[1]),c,n] = deL[i,k,c,n] + newDeL[int(pt[0]),int(pt[1]),c,n]
            return newDeL
        self.X.back(bacc)

#Node that converts a 4d tensor to a 2d tensor.
class VecFrom4D(Node):
    def __init__(self,A):
        self.A = A
        super().__init__(["A"])
    def _forward(self):
        a0,a1,c,n = self.A.fwd().shape
        return np.reshape(self.A.fwd(), [a0*a1*c,n])
    def _back(self,deL):
        def bacc():
            dell = np.reshape(deL,self.A.fwd().shape)
            return dell
        self.A.back(bacc)

#bias with 3-tensors
class ConvoNetAdd(Node):
    #params:
    #X: input
    #B:
    def __init__(self,X,B):
        # h,w,c,n
        self.X = X
        # h,w,c
        self.B = B
        super().__init__(["X","B"])
    def _forward(self):
        #get vector of size N
        V = np.ones(self.X.fwd().shape[-1])
        #expand b to number of input images;
        #broadcast addition operation over all n of them
        B2 = np.einsum("hwc,n -> hwcn", self.B.fwd(), V)
        return self.X.fwd() + B2
    def _back(self,deL):
        self.X.back(lambda: deL)
        #collapse deL over n
        self.B.back(lambda: np.sum(deL,axis=-1))
