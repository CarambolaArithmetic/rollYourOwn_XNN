try:
    import cupy as np
except ImportError:
    import numpy as np
from .math import  *

#Graph Node. Caches the value of the calculation made in the forward pass in self.value.
#   All operations inherit from this. It provides the public interface methods,
#   fwd, and back, which encapsulate the class-by-class implementation of those
#   operations following the Template pattern. This doesn't actually do
#   anything really, it's mostly just an aid to ensure that classes follow the
#   correct structure and also manage cached operations so that they aren't
#   calculated every single time a part of the network needs a forward pass from
#   one of its children.
class Node:
    #Parameters:
    #   params: A list of parameter names to finalize once when finalize() is called on this node
    #   needs_update: Whether or not this node (and thus any of its children) needs
    #       errors propogated to it in the backward pass. This would be false, for
    #       instance, if all of this node's children were input tensors and not weights.
    def __init__(self, params, needs_update = None):
        self.value = None
        self.set = False
        self.params = params
        # Calculate the grad if any of its children need it.
        if needs_update is not None:
            self.needs_update = needs_update
        else:
            self.needs_update = any([self.__dict__[p].getNeedsUpdate() for p in self.params])

    #Tells Node that backpropogation is complete.
    #   Has the side effect of clearing cached calculations for the node, which
    #   should now be useless sense backpropogation has updated the parameters.
    def finalize(self):
        self.value = None
        self.set = False
        [self.__dict__[p].finalize() for p in self.params]
    #Get the forward pass for this node.
    #   If this node has already been calculated, return that value. otherwise,
    #   calculate that value and return it
    def fwd(self):
        if self.set == False:
            self.value = self._fwd()
            self.set = True
        return self.value
    #Propogate the error to this node.
    #   get_grad: This should be a lambda that calculates the gradient from the
    #       parent, with the intent that that calculation does not need to occur
    #       unless this node actually needs it. This was done for performance
    #       reasons.
    def bck(self, get_grad):
        #Only compute the gradient if an update is actually needed
        if self.needs_update:
            grad = get_grad()
            self._bck(grad)
    #Return true if this node or any of its children have parameters that backprop needs to update.
    def getNeedsUpdate(self):
        return self.needs_update
    #Business logic for forward pass
    def _fwd(self):
        raise Exception("forward unimplemented!")
    #Business logic for backward pass
    #   grad: gradient from parent (np.ndarray, already calculated).
    def _bck(self, grad):
        raise Exception("backward unimplemented!")

#Input tensor.
#   Is not really a node, does not result from an operation, but all node
#   classes need to treat it that way because they need to assume all of their
#   children function as nodes.  Stores a numpy array as its "value", and has an
#   update rule which determines how it will update that value when given a
#   gradient in the back pass.
class Tensor:
    def __init__(self, tensor_val = np.empty([]), update_rule = lambda x, y: x - y):
        self.value = tensor_val #value returned on forward pass.
        self.grad_accumulator = self.value #Used to cache updates made on backwards pass.
        self.update_rule = update_rule

    def getNeedsUpdate(self):
        return self.update_rule != None #needs update if we have updateRule

    #Just return self.value
    def fwd(self):
        if self.value.size == 0:
            raise Exception("tensor is unset!")
        return self.value

    #Notice that this will likely be called many times during a backward pass
    #   It is the sum of these grads that provides the true gradient, not each
    #   individual one.
    def bck(self, get_grad):
        if self.grad_accumulator.size == 0:
            raise Exception("Tensor is unset!")
        if self.update_rule is not None:
            grad = get_grad()
            self.grad_accumulator = self.update_rule(self.grad_accumulator, grad)

    #Tensors often need to have their internal values changed after initialization.
    #   Example: inputs to the networks, labels.
    def set(self,tensor_val):
        self.value = tensor_val
        self.grad_accumulator = self.value
    def finalize(self):
        #Update our exposed value to reflect backward pass changes.
        self.value = self.grad_accumulator
        return

#Node that implements basic matrix multiplication
class Matmul(Node):
    #params:
    #tensor_weights of size w x m
    #tensor_input of size m x n where n is the number of inputs in the batch.
    def __init__(self, tensor_weights,tensor_input):
        self.weights = tensor_weights
        self.input = tensor_input
        super().__init__(["weights","input"])
    def _fwd(self):
        return np.matmul(self.weights.fwd(),self.input.fwd())
    def _bck(self, grad):
        self.weights.bck(lambda: np.matmul(grad,np.transpose(self.input.fwd())))
        #this could be more efficient, but I know it works. I'm transposing too many times.
        self.input.bck(lambda: np.transpose(np.matmul(np.transpose(grad),self.weights.fwd())))

#Node that implements Softmax with logistic regression.
class SoftmaxWithLogit(Node):
    #tensor_labels isn't updated. I don't know what the derivative relative to it is, nor do I care.
    #tensor_input  is m x n
    #tensor_labels is m x n
    def __init__(self, tensor_labels,tensor_input):
        self.input = tensor_input
        self.labels = tensor_labels
        self.softmax = None
        super().__init__(["labels","input"])
    def _fwd(self):
        self.softmax = np.exp(self.input.fwd())/np.sum(np.exp(self.input.fwd()),axis=0)
        return -np.sum(self.labels.fwd()*np.log(self.softmax), axis = 0)
    def _bck(self, grad):
        #only propogate to Y, because that's coming in from elsewhere in the net.
        self.input.bck(lambda: grad*(self.softmax-self.labels.fwd()))
    #Get the output of the softmax.
    def getSoftmax(self):
        return self.softmax
    #Get 1 x n vector containing predicted labels of network input
    def getPredictions(self):
        return np.argmax(self.softmax, axis = 0)

#Regular rectilinear operation.
class ReLu(Node):
    def __init__(self, tensor_input):
        self.input = tensor_input
        super().__init__(["input"])
    def _fwd(self):
        matt = self.input.fwd()
        return np.max(np.array([matt, np.zeros(matt.shape)]), axis=0)
    def _bck(self, grad):
        self.input.bck(lambda: grad * np.where(self.value > 0, 1, 0))

#Node that implements a leaky relu.
class LeakyReLu(Node):
    #params:
    #   input: m x n matrix
    def __init__(self, tensor_input):
        self.input = tensor_input
        super().__init__(["input"])
    def _fwd(self):
        matt = self.input.fwd()
        return np.max(np.array([matt, 0.1*np.ones(matt.shape)]), axis=0)
    def _bck(self, grad):
        self.input.bck(lambda: grad * np.where(self.value > 0, 1, 0.1))

#Node that implementes a *concatenation* between two inputs A and B, along axis.
#   This is useful for matmulWBias
class Cat(Node):
    #Input_a is m x n, input_b is k x n, operation returns a (m+k) x n matrix
    def __init__(self, tensor_input_a, tensor_input_b, axis):
        self.input_a = tensor_input_a
        self.input_b = tensor_input_b
        self.axis = axis
        self.splitMarker = None
        super().__init__(["input_a","input_b"])
    def _fwd(self):
        self.splitMarker = self.input_a.fwd().shape[self.axis]
        return np.concatenate([self.input_a.fwd(), self.input_b.fwd()], axis=self.axis)
    #Back pass just splits gradient into two parts, each one of which corresponds with A or B
    def _bck(self, grad):
        [gradA, gradB] = np.split(grad, [self.splitMarker], axis=self.axis)
        self.input_a.bck(lambda: gradA)
        self.input_b.bck(lambda: gradB)

#Node that implements a matrix multiplication with a bias term.
#This is the same thing as doing matmul and then an addition,
#   it just rolls the bias into the matrix operation by appending a 1 x n vector of ones along the input's m dimension
#   essentially this means we're just storing our bias weights on the end of the matmul weight matrix
class MatmulWBias(Node):
    #params: tensor_weights is weights matrix of size w x m+1
    #B is input of size m x n
    def __init__(self, tensor_weights, tensor_input, axis=0):
        self.weights = tensor_weights
        self.input = tensor_input
        self.output = None
        self.axis = axis
        #C is a result of operations on A and B, so no need to call finalize() on those
        #they get finalized when C gets finalized
        super().__init__(["output"], True)
    def _fwd(self):
        b_shape = list(self.input.fwd().shape)
        b_shape[self.axis] = 1
        one_vector = Tensor(np.ones(b_shape),None)
        #this operation can be defined as a composition of other ones we've already defined! excellent!
        self.output = Matmul(self.weights, Cat(self.input, one_vector, self.axis))
        return self.output.fwd()
    def _bck(self, grad):
        self.output.bck(lambda: grad)

#Note, only works with square inputs, as far as I know
#dims explanation:
#f: number of filters (output channels)
#c: number of channels (input channels)
#n: number of images
#l: output height and width
#Tensor operation output shape is l, l, f, n
#Designed to work in both directions(!) with strided convolution, but
#the strided part HAS NOT been tested
class Conv2D(Node):
    def __init__(self, tensor_input, tensor_kernel, stride, output_size):
        #kernel, shape: m,m,c,f
        self.kernel = tensor_kernel

        #input feature map, shape: x, x, c, n
        self.input = tensor_input

        #desired output size, needed to determine padding
        self.output_size = output_size
        self.stride = stride
        super().__init__(["kernel","input"])

    def _fwd(self):
        input_dim_0 = self.input.fwd().shape[0]
        kernel_dim_0 = self.kernel.fwd().shape[0]
        self.input_padding_size = paddingSize(kernel_dim_0, input_dim_0, self.stride, self.output_size)
        self.padded_input = pad_image(self.input.fwd(), self.input_padding_size)
        return convolve_2D(self.padded_input, self.kernel.fwd(), self.stride)

    #Convolution occurs in the backward pass of convolution, but it differs between the kernel and image
    #   when there is multiple inputs:
    #   For he kernel, there's a contraction over n, the number of input (that is, the batch) dimension
    def back_convolve_kernel(self, img, kernel, stride_length):
        submat = get_strided(img,kernel,stride_length)
        return np.einsum('hifn,klhicn->klcf', kernel, submat)

    #For the input, there is a contraction over f, the output channel dimension:
    def back_convolve_input(self, img, kernel, stride_length):
        submat = get_strided(img,kernel,stride_length)
        return np.einsum('hicf,klhifn->klcn', kernel, submat)

    #Source for the backprop rules i'm using:
    #   https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
    #   Works with strided convolution, probably.
    def _bck(self,grad):
        def XBack():
            dialated_grad = dialate(grad, self.stride-1)
            padded_grad = pad_image(dialated_grad, paddingSize(self.kernel.fwd().shape[0],
                                                     dialated_grad.shape[0],
                                                     self.stride, self.padded_input.shape[0]))
            flipped_kernel = np.flip(np.flip(self.kernel.fwd(), axis=0), axis=1)
            padded_input_grad = self.back_convolve_input(padded_grad, flipped_kernel,1)
            #gradToXPad is the gradient with respect to the PADDED version of X, so we need to clip off the parts
            #   that are with respect to the zero-padding.
            return get_clipped(padded_input_grad,self.input_padding_size)
        def KBack():
            return self.back_convolve_kernel(self.padded_input, dialate(grad, self.stride-1),1)
        self.kernel.bck(KBack)
        self.input.bck(XBack)

#Implements a max pool of a 4th order tensor which is square in two dimensions.
class SquareMaxPool(Node):
    def __init__(self,tensor_input,window_size, stride):
        self.input = tensor_input
        self.window_size = window_size
        self.stride = stride
        super().__init__(["input"])

    def _fwd(self):
        x0,x1,c,n = self.input.fwd().shape

        #Get the max of the window over input whose top corner is m0, m1.
        #   Sets the max coordinate and returns the max value with coordinates in term of X.
        def getMax(m0,m1):
            m02 = (m0+ self.window_size)
            m12 = (m1 +self.window_size)
            field = self.input.fwd()[m0:m02,m1:m12,:,:]
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
        self.num_pools = maxPoolOutSize(self.window_size, self.input.fwd().shape[0], self.stride)
        self.back_windows = np.zeros([self.num_pools,self.num_pools, 2, c, n])
        self.output = np.zeros([self.num_pools,self.num_pools, c, n])
        for i in range(0,self.num_pools):
            for k in range(0,self.num_pools):
                max_vals, arg_maxes = getMax(i*self.stride, k*self.stride)
                self.back_windows[i,k,:,:,:] = arg_maxes
                self.output[i,k,:,:] = max_vals
        return self.output
    def _bck(self, grad):
        def bacc():
            new_grad = np.zeros(self.input.fwd().shape)
            x0, x1, xc, xn = self.input.fwd().shape
            #I don't have an elegant solution to this right now, enjoy the nested for loops.
            for i in range(0,self.num_pools):
                for k in range(0, self.num_pools):
                    for c in range(0,xc):
                        for n in range(0,xn):
                            #We get the pair p0 p1 stored in the back_windows matrix that
                            #   corresponds with the output feature at i,k,c,n.
                            #   These are the h, w coords in the input for the max that was used in that output.
                            #   We then add grad[i,k,c,n] to the our new gradient tensor (newgrad) at that coord.
                            #   So if two windows have the same max, two different gradients get added to that point
                            #   in the backward tensor.
                            pt = self.back_windows[i,k,:,c,n]
                            new_grad[int(pt[0]),int(pt[1]),c,n] = grad[i,k,c,n] + new_grad[int(pt[0]),int(pt[1]),c,n]
            return new_grad
        self.input.bck(bacc)

#Node that converts a 4d tensor to a 2d tensor.
class VecFrom4D(Node):
    def __init__(self,tensor_input):
        self.input = tensor_input
        super().__init__(["input"])
    def _fwd(self):
        a0,a1,c,n = self.input.fwd().shape
        return np.reshape(self.input.fwd(), [a0*a1*c,n])
    def _bck(self,grad):
        def my_back():
            out_grad = np.reshape(grad,self.input.fwd().shape)
            return out_grad
        self.input.bck(my_back)

#Bias with 3-tensors.
class ConvoNetAdd(Node):
    #params:
    #   tensor_input: dimensions height x width x channels x batch (h,w,c,n)
    #   tensor_bias: dimensions height x width x channels (h,w,c)
    def __init__(self,tensor_input,tensor_bias):
        self.input = tensor_input
        self.bias = tensor_bias
        super().__init__(["input","bias"])
    def _fwd(self):
        #Get vector of size n
        ones_vector = np.ones(self.input.fwd().shape[-1])
        #Broadcast addition operation over n
        bias_broadcasted = np.einsum("hwc,n -> hwcn", self.bias.fwd(), ones_vector)
        return self.input.fwd() + bias_broadcasted
    def _bck(self,grad):
        self.input.bck(lambda: grad)
        #collapse grad over n
        self.bias.bck(lambda: np.sum(grad,axis=-1))
