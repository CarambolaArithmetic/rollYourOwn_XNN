try:
        import cupy as np
except ImportError:
        import numpy as np

#Padded an array? unpad it with THIS!
#   note: must be square in the first two dims
def get_clipped(square_arr_4d,reduce_by):
    x0 = square_arr_4d.shape[0]
    return square_arr_4d[reduce_by:x0-reduce_by,reduce_by:x0-reduce_by,:,:]

#Generates a tensor of shape [k,l, h,i,c,n]
#   Which varies by channel over c
#   Varies by input over n
#   Gives an hxi convolutional field that varies over the output dimensions k and l
#   That is, k and l are the output image height and width
#   This is a tricky memory operation, but trust me, i've tested THIS bit thoroughly
def get_strided(img,kernel,s):
    strided = np.lib.stride_tricks.as_strided
    s0,s1,s2,s3 = img.strides
    mi,hi,c,n = img.shape
    mk,hk = kernel.shape[0:2]
    out_shp  = (1+(mi-mk)//s, 1+(hi-hk)//s, mk, hk, c, n)
    return strided(img, shape=out_shp, strides=(s*s0,s*s1,s0,s1,s2,s3))

#Convolve image with kernel using the provided stridelength.
def convolve_2D(img, kernel, stride_length):
    submat = get_strided(img,kernel,stride_length)
    return np.einsum('hicf,klhicn->klfn', kernel, submat)

#k = kernel size
#i = input size
#s = strides
#l = desired output size
#Note that this probably isn't entirely correct, there should be a floor/ceiling function in here, but
#   Python will alert you if the sizes are wrong.
def paddingSize(k, i, s, l):
    return int((s*(l-1)+k-i)/2)

def maxPoolOutSize(k,i,s):
    return int((i-k)/s + 1)

#Zero-pad a 4-tensor image set with a layer of p zeros on every side.
def pad_image(img, p):
    return np.pad(img,((p,p),(p,p),(0,0),(0,0)))

#Dialates but clips the zeros off the bottom and right.
# Useful for finding the gradient of a strided convolution, which is fortunately not used.
def dialate(img, dialate_amount, do_clip = True):
    clip = 0
    if do_clip:
        clip = dialate_amount
    dA = dialate_amount
    s0, s1, s2,s3 = img.shape
    X = np.zeros([s0*(dA+1)-clip, s1*(dA+1)-clip, s2,s3])
    for h in range(0,s0):
        for v in range(0,s1):
            X[(dA+1)*h,(dA+1)*v,:,:] = img[h,v,:,:]
    return X
