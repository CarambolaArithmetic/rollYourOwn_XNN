try:
        import cupy as np
except ImportError:
        import numpy as np

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
