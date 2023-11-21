################################################################################
# CREDIT:
#    This demo was created as an assignment submission for Arthur Redfern's
#    CS6301 Convolutional Neural Networks course. It is based off of a template,
#    substantially modified by myself. The source for the template can be found
#    at:
#    https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/blob/master/Tests/202008/xNNs_Project_001_Math.py
#
# INSTRUCTIONS
#
#    This script takes no arguments, when run and it will download training data and begin training.
#
# DESCRIPTION
#   This file implements a basic mutlilayer neural network.
#   It uses a flat learning rate, with stochastic batches of 100 images selected at each iteration via sample-with-replacement.
#   In order to improve training speed, the network uses different learning rates for each layer,
#   with lower layers having higher learning rates to speed up their updates while still allowing the overall
#   learning rate to be small to avoid divergence.
#
#   The Network definition relies on a set of classes which each implement
#   a forward/backward definition for the associated operations
#   and are capable of backpropagating their error to their children. This allows easy, dynamic implementation
#   of networks in a fashion that looks very similar to composing functions, which can be seen in the
#   genNetwork() function in the HELPER FUNCTIONS section as an example.
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
DATA_NUM_TRAIN = 60000
DATA_NUM_TEST = 10000
DATA_CHANNELS = 1
DATA_ROWS = 28
DATA_COLS = 28
DATA_CLASSES = 10
DATA_URL_TRAIN_DATA = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA = 'test_data.gz'
DATA_FILE_TEST_LABELS = 'test_labels.gz'

# display
DISPLAY_ROWS = 8
DISPLAY_COLS = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM = DISPLAY_ROWS*DISPLAY_COLS

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
if (os.path.exists(DATA_FILE_TRAIN_DATA) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data = frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels = frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data = frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels = frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

################################################################################
# HELPER FUNCTIONS###############################################################

# Accuracy measurement. One hot labels because its convenient,
#   not because its efficient


def accuracy(one_hot_labels, integer_class_predictions):
    one_if_true = np.where(
        np.equal(np.argmax(one_hot_labels, axis=0), integer_class_predictions), 1, 0)
    return 100*sum(one_if_true)/one_if_true.shape[0]

# Returns accuracy of network on all testing data.
#   Calculate in 100 data increments because excessive size is a problem
#   for some things that i have implemented (convolution, specifically, einsum doesn't like big args)
#   and I wanted it to work for both.


def allDataAccuracy(network, input_tensor, label_tensor):
    acc = []
    subacc_size = 100
    for i in range(0, DATA_NUM_TEST//subacc_size):
        network.finalize()
        start = i * subacc_size
        end = (i + 1) * subacc_size
        labels = np.eye(DATA_CLASSES)[test_labels[start:end]]

        inputs = test_data[start:end]
        ready_labels, ready_inputs = preproc(inputs, labels)
        input_tensor.set(ready_inputs)
        label_tensor.set(ready_labels)

        # calculate outputs and get predictions
        network.get_net_head().fwd()
        predictions = network.get_net_head().getPredictions()

        acc.append(accuracy(ready_labels, predictions))
    return np.mean(np.array(acc))

# Generates the standard network based the defined pattern of
#   matmulWBias->relu-matmulWBias->relu
# Params:
#   input_tensor: the input of the network (unset)
#   labels_tensor: values to compare to (unset)
#   sizes: dims of various parts of the network.


def genNetwork(input_tensor, labels_tensor, sizes):
    network = input_tensor
    params_list = []
    num_layers = len(sizes)
    for i in range(1, num_layers-1):
        # In addition to the learning rate, we have a layer-specific factor to improve training speed.
        learning_rate_factor = 3*(num_layers-i)+1
        # Because we're using relus, values in the network are prone to exploding
        #   It's best if we start with very very small weights so we don't get overflow errors.
        W = Tensor(0.01*np.random.rand(sizes[i], sizes[i-1]+1),
                   update_rule=lambda x, y: x-learning_rate_factor*y)
        params_list.append(W)
        network = LeakyReLu(MatmulWBias(W, network))
    W = Tensor(0.01*np.random.rand(sizes[num_layers-1],
               sizes[num_layers-2]+1), update_rule=lambda x, y: x-y)
    network = MatmulWBias(W, network)
    network = ErrorWithNormalizationTerms(
        SoftmaxWithLogit(labels_tensor, network), 0.02)
    # Network actually works better without this, but it does work
    #   network.addNormalizationParams(paramsList)
    return network

# Trains the network passed as the network arg.
# params:
#   network: network to train
#   learning_rate: the learning rate
#   do_tensor_update: updates input and label tensor
#   labels_tensor: the labels tensor


def trainNetwork(network, learning_rate, num_epochs, do_tensor_update, labels_tensor, inputs_tensor):
    learning_data = {}
    learning_data["batchErr"] = []
    learning_data["epochAccs"] = []
    learning_data["time"] = 0
    learning_data["batchAccs"] = [[], []]
    learning_data["allAccs"] = [[], []]
    # Measure of how strong predictions are. This should increase as training goes along --
    #   decreasing means something is wrong; that the classifier is regressing to the mean.
    #   This tends to happen if the backprop is wrong.

    def certainty(arra):
        return 100 * np.average((np.max(arra, axis=0)/np.sum(arra, axis=0)))
    i = 0
    start_time = time.perf_counter()
    old_epoch_end_time = start_time
    print("Starting training...")
    for j in range(num_epochs):
        for k in range(DATA_NUM_TRAIN):

            # update input/label tensors
            do_tensor_update()

            if i % 200 == 0:
                print("ERROR at iteration " + str(i) +
                      ": " + str(network.pure_error()))
                print("    certainty: " +
                      str(certainty(network.get_net_head().getSoftmax())))
                batch_accuracy = accuracy(
                    labels_tensor.fwd(), network.get_net_head().getPredictions())
                learning_data["batchAccs"][0].append(i)
                learning_data["batchAccs"][1].append(batch_accuracy)
                print("    accuracy on batch: " + str(batch_accuracy))

            learning_data["batchErr"].append(network.pure_error())

            # Update the weights.
            network.propogate_error(learning_rate)

            # Tell network nodes to finalize their cached values, or at least replace them with a new value
            #   next time they are needed.
            network.finalize()
            if i % 2000 == 0:
                all_accuracy = allDataAccuracy(
                    network, inputs_tensor, labels_tensor)
                learning_data["allAccs"][0].append(i)
                learning_data["allAccs"][1].append(all_accuracy)
                print("---")
                print("Test Accuracy " + str(all_accuracy))
                print("---")
                network.finalize()
            i = i+1
        acc = allDataAccuracy(network, inputs_tensor, labels_tensor)
        print()
        print("accuracy after epoch " + str(j) + ": " + str(acc) + "%")
        epoch_end_time = time.perf_counter()
        print("epoch took " + str(epoch_end_time - old_epoch_end_time) + " seconds")
        print()
        old_epoch_end_time = epoch_end_time
        learning_data["epochAccs"].append(acc)

    # Set end timer, store value for display.
    end_time = time.perf_counter()
    learning_data["time"] = end_time-start_time
    print("total train time time: " + str(end_time-start_time) + " seconds.")
    return learning_data

# Preprocess the input and output data.


def preproc(inputs, labels):
    labels = np.transpose(labels)
    inputs = np.transpose(inputs)
    # Flatten.
    inputs = (1/255)*inputs.reshape(-1, *inputs.shape[-1:])
    return (labels, inputs)

# Generate a batch from the set of inputs and labels using sample-with-replacement


def genBatch(batch_size, inputs, labels):
    low = 0
    high = inputs.shape[0]-1
    nums = np.random.randint(low=low, high=high, size=batch_size)
    output_labels = np.eye(DATA_CLASSES)[labels[nums]]
    output_inputs = inputs[nums, :, :]
    return preproc(output_inputs, output_labels)

# Set the values of the tensors A and B.
#   In our case, we just use this to set the input and label tensors.


def updateTensors(A, B, ABList):
    A.set(ABList[0])
    B.set(ABList[1])

# END HELPER FUNCTIONS###########################################################
################################################################################
# TRAINING#######################################################################

def demo():
    # create our "placeholder" tensors:
    # inputs, does not get backpropped
    X = Tensor(update_rule=None)

    # L for labels, also does not have a backprop rule
    L = Tensor(update_rule=None)

    # The network definition. see "genNetwork" function above:
    network_layers = [784, 1000, 100, 10]
    batch_size = 100
    network = genNetwork(X, L, network_layers)

    # do the training, using a batch size of 100
    # 2 "epochs" (not actual epochs, just 12000 iterations of batch training).
    # 0.001 learning rate really only works for this batch size smaller batches need smaller rates to prevent exploding.
    # A dynamic learning rate would probably improve this, but there wasn't time to experiment.
    train_specs = trainNetwork(network, 0.001,
                            1,
                            do_tensor_update=lambda: updateTensors(
                                L, X, genBatch(batch_size, train_data, train_labels)),
                            labels_tensor=L,
                            inputs_tensor=X)
