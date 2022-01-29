################################################################################
# INSTRUCTIONS
#
#    <TODO: provide correct instructions here>
#
#DESCRIPTION
#  This file implements a basic mutlilayer neural network.
#   it uses a flat learning rate, with stochastic batches of 100 images selected at each iteration via sample-with-replacement.
#   in order to improve training speed, the network uses different learning rates for each layer,
#   with lower layers having higher learning rates to speed up their updates while still allowing the overall
#   learning rate to be small to avoid divergence.
#
#  The Network definition relies on a set of classes which each implement
#  a forward/backward definition for the associated operations
#  and are capable of backpropagating their error to their children.
#  this can be seen in the OPERATION DEFINITIONS section below. this allows easy, dynamic implementation
#  of networks in a fashion that looks very similar to composing functions, which can be seen in the
#  genNetwork() function in the HELPER FUNCTIONS section as an example
#
#   After the operation section is a set of helper functions used for the purpose of training the net.
#   see the section  HELPER FUNCTIONS  below for these functions.
#
#   This is followed by a "training" section which uses the logic from the two sections above
#   to build and train the network.
#
#   A note about dimensions:
#       when doing the forward pass, I prefer to display matrix operations  with the input vector on the right sideo.
#       because everything uses batch, this means the input will follow
#       the convention of being m x n, where m spans the data for a single image and n is the number of images in the batch
#       forward pass functions are written using this convention,
#       so matmul(W,X) represents WxX where W is the weight matrix and X is the set of inputs.
#       The only exceptions would be where I forgot to do this, which I *hope* is nowhere.
#
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
DATA_NUM_TEST          = 10000
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

#END OP DEFINITIONS#############################################################
################################################################################
#HELPER FUNCTIONS###############################################################

#accuracy measurement. one hot labels because its convenient,
#not because its efficient
def accuracy(oneHotLabels, integerClassPredictions):
    oneIfTrue = np.where(np.equal(np.argmax(oneHotLabels, axis = 0), integerClassPredictions), 1, 0)
    return 100*sum(oneIfTrue)/oneIfTrue.shape[0]

#returns accuracy of network on all testing data
#calculate in 100 data increments because excessive size is a problem
# for some things that i have implemented (convolution, specifically, einsum doesn't like big args)
# and I wanted it to work for both
def allDataAccuracy(network, inputTensor, labelTensor):
        acc = []
        subaccSize = 100
        for i in range(0, DATA_NUM_TEST//subaccSize):
            network.clear()
            start = i * subaccSize
            end = (i + 1) * subaccSize
            labels = np.eye(DATA_CLASSES)[test_labels[start:end]]

            inputs = test_data[start:end]
            readyLabels, readyInputs = preproc(inputs, labels)
            inputTensor.set(readyInputs)
            labelTensor.set(readyLabels)

            #calculate outputs and get predictions
            network.getNetHead().fwd()
            predictions = network.getNetHead().getPredictions()

            acc.append(accuracy(readyLabels, predictions))
        return np.mean(acc)

#generates the standard network based the defined pattern of
#matmulWBias->relu-matmulWBias->relu
#params:
# inputTensor: the input of the network (unset)
# labelsTensor: values to compare to (unset)
# sizes: dims of various parts of the network.
def genNetwork(inputTensor, labelsTensor, sizes):
    network = inputTensor
    paramsList = []
    numLayers =  len(sizes)
    for i in range(1, numLayers-1):
        #In addition to the learning rate, we have a layer-specific factor to improve training speed.
        learningRateFactor = 3*(numLayers-i)+1
        #because we're using relus, values in the network are prone to exploding
        #it's best if we start with very very small weights so we don't get overflow errors
        W = Tensor(0.01*np.random.rand(sizes[i], sizes[i-1]+1), updateRule = lambda x, y: x-learningRateFactor*y)
        paramsList.append(W)
        network = LeakyReLu(MatmulWBias(W,network))
    W = Tensor(0.01*np.random.rand(sizes[numLayers-1], sizes[numLayers-2]+1), updateRule = lambda x, y: x-y)
    network = MatmulWBias(W,network)
    network = ErrorWithNormalizationTerms(SoftmaxWithLogit(labelsTensor, network), 0.02)
    #network actually works better without this, but it does work
    #network.addNormalizationParams(paramsList)
    return network

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
        for k in range(DATA_NUM_TRAIN):

            #update input/label tensors
            doTensorUpdate()

            if i%200 == 0:
                print("ERROR at iteration " + str(i) + ": " + str(network.pureError()))
                print("    certainty: " + str(certainty(network.getNetHead().getSoftmax())))
                batchAccuracy = accuracy(labelsTensor.fwd(),network.getNetHead().getPredictions())
                learningData["batchAccs"][0].append(i)
                learningData["batchAccs"][1].append(batchAccuracy)
                print("    accuracy on batch: " + str(batchAccuracy))

            learningData["batchErr"].append(network.pureError())

            #update the weights.
            network.propogateError(learningRate)

            #tell network nodes to clear their cached values, or at least replace them with a new value
            #next time they are needed.
            network.clear()
            if i%2000 == 0:
                allAccuracy = allDataAccuracy(network, inputsTensor, labelsTensor)
                learningData["allAccs"][0].append(i)
                learningData["allAccs"][1].append(allAccuracy)
                print("---")
                print("Test Accuracy " + str(allAccuracy))
                print("---")
                network.clear()
            i = i+1
        acc = allDataAccuracy(network, inputsTensor, labelsTensor)
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

#preprocess the input and output data
def preproc(inputs, labels):
    labels = np.transpose(labels)
    inputs = np.transpose(inputs)
    #flatten
    inputs = (1/255)*inputs.reshape(-1, *inputs.shape[-1:])
    return (labels, inputs)

# Generate a batch from the set of inputs and labels using sample-with-replacement
def genBatch(batchSize, inputs, labels):
    low=0
    high= inputs.shape[0]-1
    nums = np.random.randint(low=low, high=high, size=batchSize)
    outputLabels = np.eye(DATA_CLASSES)[labels[nums]]
    outputInputs = inputs[nums,:,:]
    return preproc(outputInputs, outputLabels)

#set the values of the tensors A and B.
#in our case, we just use this to set the input and label tensors
def updateTensors(A,B, ABList):
    A.set(ABList[0])
    B.set(ABList[1])

#END HELPER FUNCTIONS###########################################################
################################################################################
#TRAINING#######################################################################

#create our "placeholder" tensors:
#inputs, does not get backpropped
X = Tensor(updateRule = None)

#L for labels, also does not have a backprop rule
L = Tensor(updateRule = None)

#The network definition. see "genNetwork" function above:
networkLayers = [784,1000,100,10]
batchSize = 100
network = genNetwork(X, L, networkLayers)

#do the training, using a batch size of 100
#2 "epochs" (not actual epochs, just 12000 iterations of batch training).
# 0.001 learning rate really only works for this batch size smaller batches need smaller rates to prevent exploding.
#A dynamic learning rate would probably improve this, but there wasn't time experiment.
trainSpecs = trainNetwork(network, 0.001,
             1,
             doTensorUpdate = lambda: updateTensors(L,X,genBatch(batchSize,train_data,train_labels)),
             labelsTensor = L,
             inputsTensor = X)
