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
try:
        import cupy as np
except ImportError:
        import numpy as np
import numpy
from ryoxnn.node import *
from ryoxnn.network import *

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
def frombuffer(*args, **kwargs):
    """Interpret a buffer as a 1-dimensional array.
    .. note::
        Uses NumPy's ``frombuffer`` and coerces the result to a CuPy array.
    .. seealso:: :func:`numpy.frombuffer`
    """
    return np.asarray(numpy.frombuffer(*args, **kwargs))

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
train_data        = frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

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
    acc_arr = np.array(acc)
    print("ACCURACY: " + str(np.mean(acc_arr)))
    return np.mean(acc_arr)

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
