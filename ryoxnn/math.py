try:
    import cupy as np
except ImportError:
    import numpy as np


def get_clipped(square_arr_4d, reduce_by):
    """
    Unpad a padded array. input must be square in the first two dims.
    """
    x0 = square_arr_4d.shape[0]
    return square_arr_4d[reduce_by:x0-reduce_by, reduce_by:x0-reduce_by, :, :]


def get_strided(img, kernel, s):
    """
    Generates a tensor of shape [k,l, h,i,c,n],
    Which varies by channel over c,
    Varies by input over n,
    Gives an hxi convolutional field that varies over the output dimensions k and l
    That is, k and l are the output image height and width.

    This a memory operation which simply returns a view to the original data.
    """
    strided = np.lib.stride_tricks.as_strided
    s0, s1, s2, s3 = img.strides
    mi, hi, c, n = img.shape
    mk, hk = kernel.shape[0:2]
    out_shp = (1+(mi-mk)//s, 1+(hi-hk)//s, mk, hk, c, n)
    return strided(img, shape=out_shp, strides=(s*s0, s*s1, s0, s1, s2, s3))


def convolve_2D(img, kernel, stride_length):
    """
    Convolve image with kernel using the provided stridelength.
    """
    submat = get_strided(img, kernel, stride_length)
    return np.einsum('hicf,klhicn->klfn', kernel, submat)


def paddingSize(k, i, s, l):
    """
    Parameters
    __________
    k : 
        kernel size
    i : 
        input size
    s : 
        strides
    l : 
        desired output size
        TODO: there might need to be a floor/ceiling function in here, but
            Python will alert user if the sizes are wrong.
    """
    return int((s*(l-1)+k-i)/2)


def maxPoolOutSize(k, i, s):
    return int((i-k)/s + 1)


def pad_image(img, p):
    """
    Zero-pad a 4-tensor image set with a layer of p zeros on every side.
    """
    return np.pad(img, ((p, p), (p, p), (0, 0), (0, 0)))


def dialate(img, dialate_amount, do_clip=True):
    """
    Dialates but clips the zeros off the bottom and right.
    Useful for finding the gradient of a strided convolution.
    """
    clip = 0
    if do_clip:
        clip = dialate_amount
    dA = dialate_amount
    s0, s1, s2, s3 = img.shape
    X = np.zeros([s0*(dA+1)-clip, s1*(dA+1)-clip, s2, s3])
    for h in range(0, s0):
        for v in range(0, s1):
            X[(dA+1)*h, (dA+1)*v, :, :] = img[h, v, :, :]
    return X
