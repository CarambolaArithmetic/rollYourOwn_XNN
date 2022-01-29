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
