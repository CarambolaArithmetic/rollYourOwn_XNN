################################################################################
# INSTRUCTIONS
#
#    <TODO: provide correct instructions here>
#
#DESCRIPTION
#  This file implements a Convolutional neural network.
#   it uses a flat learning rate, with stochastic batches of 2 images selected at each iteration via sample-with-replacement.
#   in order to improve training speed, the network uses different learning rates for each layer,
#   with lower layers having higher learning rates to speed up their updates while still allowing the overall
#   learning rate to be small to avoid divergence. The batch size is so small because the convolution operation takes
#   so much time and larger sized input tensors are rejected by numpy
#
#  Like the NN in part 1, the network definition relies on a set of classes which each implement
#  a forward/backward definition for the associated operations and are capable of backpropogating their error to their children.
#  all of the operations from part 1 are included here. Additionally included are several operations specific
#  to convolution, including the Convolution, SquareMaxpool and VecFrom4D operations.
#
#  this file is structured the same as part 1, with operations, then helper functions, then the part where training occurs.
#   A note about dimensions:
#       inputs to the convo net are 4th order tensors, of the form:
#           h x w x c x n
#           where h is the height of the image, w is the width, c is the channel dimension, and n is the number of inputs
#       the inputs to each convo layer follow this pattern.
#       Kernels are also 4th order tensors, of the form:
#           h x w x c x f
#           where h and w are height an width, c is the input channel dim, and f is the output channel dim.
################################################################################

################################################################################
#
# IMPORT
#
################################################################################
import os.path
import urllib.request
import gzip
import math
import numpy             as np

import time

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 100
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

#OPERATION DEFINITIONS##########################################################

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
        return np.max([matt, np.zeros(matt.shape)], axis=0)
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
        return np.max([matt, 0.1*np.ones(matt.shape)], axis=0)
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


#padded an array? unpad it with THIS!
#note: must be square in the first two dims
def getClipped(squareArr4D,reduceBy):
    x0 = squareArr4D.shape[0]
    return squareArr4D[reduceBy:x0-reduceBy,reduceBy:x0-reduceBy,:,:]

#generates a tensor of shape [k,l, h,i,c,n]
#which varies by channel over c
#varies by input over n
#gives an hxi convolutional field that varies over the output dimensions k and l
#that is, k and l are the output image height and width
#this is a tricky memory operation, but trust me, i've tested THIS bit thoroughly
def getStrided(img,kernel,s):
    strided = np.lib.stride_tricks.as_strided
    s0,s1,s2,s3 = img.strides
    mi,hi,c,n = img.shape
    mk,hk = kernel.shape[0:2]
    out_shp  = (1+(mi-mk)//s, 1+(hi-hk)//s, mk, hk, c, n)
    return strided(img, shape=out_shp, strides=(s*s0,s*s1,s0,s1,s2,s3))

#convolve image with kernel using the provided stridelength
def convoolv3(img, kernel, strideLength):
    submat = getStrided(img,kernel,strideLength)
    return np.einsum('hicf,klhicn->klfn', kernel, submat)

#Convolution occurs in the backward pass of convolution, but it differs between the kernel and image
#when there is multiple inputs:
#for K (the kernel), there's a contraction over n, the number of input dimension
def backConvoolvK(img, kernel, strideLength):
    submat = getStrided(img,kernel,strideLength)
    return np.einsum('hifn,klhicn->klcf', kernel, submat)

#for X, there is a contraction over f, the output channel dimension
def backConvoolvX(img, kernel, strideLength):
    submat = getStrided(img,kernel,strideLength)
    return np.einsum('hicf,klhifn->klcn', kernel, submat)

#k = kernel size
#i = input size
#s = strides
#l = desired output size
#note that this probably isn't entirely correct, there should be a floor/ceiling function in here, but
#python will alert you if the sizes are wrong
def paddingSize(k, i, s, l):
    return int((s*(l-1)+k-i)/2)
def maxPoolOutSize(k,i,s):
    return int((i-k)/s + 1)

# zero-pad a 4-tensor image set with a layer of p zeros on every side.
def padImage(img, p):
    return np.pad(img,((p,p),(p,p),(0,0),(0,0)))

#dialates but clips the zeros off the bottom and right
#useful for finding the gradient of a strided convolution, which is fortunately not used.
def dialate(img, dialateAmount, doClip = True):
    clip = 0
    if doClip:
        clip = dialateAmount
    dA = dialateAmount
    s0, s1, s2,s3 = img.shape
    X = np.zeros([s0*(dA+1)-clip, s1*(dA+1)-clip, s2,s3])
    for h in range(0,s0):
        for v in range(0,s1):
            X[(dA+1)*h,(dA+1)*v,:,:] = img[h,v,:,:]
    return X

#END OP DEFINITIONS#############################################################
################################################################################
#HELPER FUNCTIONS###############################################################

#accuracy measurement. one hot labels because its convenient,
#not because its efficient
def accuracy(oneHotLabels, integerClassPredictions):
    oneIfTrue = np.where(np.equal(np.argmax(oneHotLabels, axis = 0), integerClassPredictions), 1, 0)
    return 100*sum(oneIfTrue)/oneIfTrue.shape[0]


#trains the network passed as the network arg.
#params:
# network: network to train
# learningRate: the learning rate
# shouldStop: lambda(x), true if training should stop
# doTensorUpdate: updates input and label tensor
# labelsTensor: the labels tensor
def trainNetwork(network, learningRate, numEpochs, doTensorUpdate, labelsTensor, inputsTensor):
    learningData = {}
    learningData["batchErr"] = []
    learningData["epochAccs"] = []
    learningData["time"] = 0
    learningData["batchAccs"] = [[],[]]
    learningData["allAccs"] = [[],[]]
    #measure of how strong predictions are. this should increase as training goes along
    #decreasing means something is wrong; that the classifier is regressing to the mean
    #this tends to happen if the backprop is wrong.
    def certainty(arra):
        return 100 * np.average((np.max(arra, axis = 0)/np.sum(arra, axis = 0)))
    i = 0
    startTime = time.perf_counter()
    oldEpochEndTime = startTime
    print("starting training...")
    for j in range(numEpochs):
        for k in range(DATA_NUM_TRAIN//4):

            #update input/label tensors
            doTensorUpdate()
            print("ERROR at iteration " + str(i) + ": " + str(network.pureError()))
            if i%200 == 0:
                print("    certainty: " + str(certainty(network.getNetHead().getSoftmax())))
                batchAccuracy = accuracy(labelsTensor.fwd(),network.getNetHead().getPredictions())
                learningData["batchAccs"][0].append(i)
                learningData["batchAccs"][1].append(batchAccuracy)
                print("    accuracy on batch: " + str(batchAccuracy))

            learningData["batchErr"].append(network.pureError())

            #update the weights.
            network.propogateError(learningRate(i))

            #tell network nodes to clear their cached values, or at least replace them with a new value
            #next time they are needed.
            network.clear()
            if i+1%10000 == 0:
                allAccuracy = checkConvoAccuracy(network, X,L, test_data,test_labels)
                learningData["allAccs"][0].append(i)
                learningData["allAccs"][1].append(allAccuracy)
                print("---")
                print("Test Accuracy " + str(allAccuracy))
                print("---")
                network.clear()
            i = i+1
        acc = checkConvoAccuracy(network, X,L, test_data,test_labels)
        print()
        print("accuracy after epoch " + str(j) + ": " + str(acc) + "%")
        epochEndTime = time.perf_counter()
        print("epoch took " + str(epochEndTime - oldEpochEndTime) + " seconds")
        print()
        oldEpochEndTime = epochEndTime
        learningData["epochAccs"].append(acc)

    #set end timer, store value for display
    endTime = time.perf_counter()
    learningData["time"] = endTime-startTime
    print("total train time time: " + str(endTime-startTime) + " seconds.")
    return learningData


#set the values of the tensors A and B.
#in our case, we just use this to set the input and label tensors
def updateTensors(A,B, ABList):
    A.set(ABList[0])
    B.set(ABList[1])

def genConvoBatch(batchSize, inputs, labels):
    low=0
    high= inputs.shape[0]-1
    nums = np.random.randint(low=low, high=high, size=batchSize)
    outputLabels = np.eye(DATA_CLASSES)[labels[nums]]
    outputLabels = np.transpose(outputLabels)
    outputInputs = np.transpose(inputs[nums,:,:])
    outputInputs = (1/255)*outputInputs
    return [outputLabels, outputInputs]

def genTestCNNNet(inputs, labels):
    #significantly increasing the learning rates of the lower layers improves training speed,
    #clipping the gradient imporoves network stability (which is terrible in general)
    CN1 = Tensor(0.01*np.random.rand(3,3,1,32), updateRule = lambda x, y: x-np.clip(1000*y,-0.0001,0.0001))
    B1 = Tensor(0.01*np.random.rand(28,28,32), updateRule = lambda x, y: x-np.clip(1000*y,-0.0001,0.0001))

    CN2 = Tensor(0.01*np.random.rand(3,3,32,64), updateRule = lambda x, y: x-np.clip(100*y,-0.0001,0.0001))
    B2 = Tensor(0.01*np.random.rand(15,15,64), updateRule = lambda x, y: x-np.clip(100*y,-0.0001,0.0001))

    W1 = Tensor(0.01 * np.random.rand(10, 3137), updateRule=lambda x, y: x - np.clip(0.1*y,-0.0001,0.0001))

    acc = Convolve(inputs, CN1,1,28)
    acc = ConvoNetAdd(acc, B1)
    acc = ReLu(acc)
    acc = SquareMaxPool(acc,3,2)

    acc = Convolve(acc, CN2,1,15)
    acc = ConvoNetAdd(acc, B2)
    acc = ReLu(acc)
    acc = SquareMaxPool(acc,3,2)

    acc = VecFrom4D(acc)

    acc = MatmulWBias(W1, acc)
    acc = SoftmaxWithLogit(labels, acc)

    network = ErrorWithNormalizationTerms(acc, 0.02)
    return network

#for 10000: 0.0375
def genCNNNet(inputs,labels):
    CN1 = Tensor(0.01*np.random.rand(3,3,1,16), updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))
    B1 = Tensor(0.01*np.random.rand(28,28,16), updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))

    CN2 = Tensor(0.01*np.random.rand(3,3,16,32), updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))
    #todo: should be 14,14,32
    B2 = Tensor(0.01*np.random.rand(15,15,32), updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))

    CN3 = Tensor(0.01*np.random.rand(3,3,32,64),  updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))
    B3 = Tensor(0.01*np.random.rand(7,7,64),  updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))

    #why 3137 and 101 instead of 3136 and 100? because bias
    W1 = Tensor(0.01*np.random.rand(100,3137), updateRule = lambda x, y: x-np.clip(y,-0.0001,0.0001))
    W2 = Tensor(0.01*np.random.rand(10,101), updateRule = lambda x, y: x-np.clip(y, -0.0001,0.0001))

    #acc = accumulator
    acc = Convolve(inputs,CN1,1,28)
    acc = ConvoNetAdd(acc, B1)
    acc = ReLu(acc)
    acc = SquareMaxPool(acc,3,2)

    acc = Convolve(acc, CN2,1,15)
    acc = ConvoNetAdd(acc, B2)
    acc = ReLu(acc)
    acc = SquareMaxPool(acc,3,2)

    acc = Convolve(acc, CN3,1,7)
    acc = ConvoNetAdd(acc, B3)
    acc = ReLu(acc)

    acc = VecFrom4D(acc)

    acc = MatmulWBias(W1, acc)
    acc = ReLu(acc)
    acc = MatmulWBias(W2, acc)

    acc = SoftmaxWithLogit(labels, acc)
    network = ErrorWithNormalizationTerms(acc, 0.02)
    return network

def convoPreproc(inputs, labels):
    labels = np.transpose(labels)
    inputs = np.transpose(inputs)
    inputs = (1/255)*inputs
    return (labels, inputs)

def checkConvoAccuracy(network, mX, mL, datas, labels):
    acc = []
    subaccSize = 100
    for i in range(0, DATA_NUM_TEST//subaccSize):
        network.clear()
        start = i * subaccSize
        end = (i + 1) * subaccSize
        print("calculating accuracy for " + str(i*subaccSize))
        outputLabels = np.eye(DATA_CLASSES)[labels[start:end]]
        outputInputs = datas[start:end]
        outputLabels, outputInputs = convoPreproc(outputInputs, outputLabels)
        mX.set(outputInputs)
        mL.set(outputLabels)
        network.getNetHead().fwd()
        predictions = network.getNetHead().getPredictions()
        acc.append(accuracy(outputLabels, predictions))
    print("ACCURACY: " + str(np.mean(acc)))
    return np.mean(acc)

#END HELPER FUNCTIONS###########################################################
################################################################################
#TRAINING#######################################################################

#create our "placeholder" tensors:
#inputs, does not get backpropped
X = Tensor(updateRule = None)

#L for labels, also does not have a backprop rule
L = Tensor(updateRule = None)


network = genCNNNet(X,L)

def learningRate(i):
    a = 0.1/(1+0.0001*i)
    print(a)
    return a
trainSpecs = trainNetwork(network, learningRate,
             1,
             doTensorUpdate = lambda: updateTensors(L,X, genConvoBatch(2,train_data,train_labels)),
             labelsTensor = L,
             inputsTensor = X)
checkConvoAccuracy(network, X, L, test_data, test_labels)
