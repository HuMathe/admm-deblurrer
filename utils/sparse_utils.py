import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from numpy.lib.stride_tricks import sliding_window_view

def sparseConvMat(img_shape, filt_kernel):
    kh, kw = filt_kernel.shape
    ph, pw = kw//2, kw//2
    pindices = np.arange(np.prod(img_shape)).reshape(img_shape)
    convidx_windows = sliding_window_view(np.pad(pindices, (ph, pw), 'reflect'), (kh, kw))

    col_indices = convidx_windows.ravel()
    row_indices = np.repeat(np.arange(np.prod(img_shape))[..., None], kh*kw, 1).ravel()
    data = np.repeat(filt_kernel.ravel()[None,...], np.prod(img_shape), 0).ravel()
    conv_mat = sparse.coo_matrix((data, (row_indices, col_indices)), (np.prod(img_shape), np.prod(img_shape)))
    return conv_mat 