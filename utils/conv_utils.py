import numpy as np


def make_conv_func(kernel, mode, with_gramian=True):

    conv_func = None
    gramian_func = None

    _kernelic = np.copy(kernel)
    _kernelic_rev = np.copy(kernel)[::-1,::-1]
    
    if mode == 'cv2':

        import cv2
        conv_func = lambda image: cv2.filter2D(image, -1, _kernelic)
        gramian_func = lambda image: cv2.filter2D(cv2.filter2D(image, -1, _kernelic), -1, _kernelic_rev)

    elif mode == 'fft':

        from utils.fft_utils import convFilterFunc, convGramianFunc
        conv_func = convFilterFunc(_kernelic)
        gramian_func = convGramianFunc(_kernelic)

    elif mode == 'scipy':

        from scipy.signal import convolve2d
        conv_func = lambda image: convolve2d(image, _kernelic, boundary='symm', mode='same')
        gramian_func = (lambda image: 
                        convolve2d(
                            convolve2d(image, _kernelic, boundary='symm', mode='same'), 
                            _kernelic_rev, boundary='symm', mode='same'))

    elif mode == 'dda4310whl':
        from simple_conv2d import conv2d_fp64
        pad_n_conv = lambda image, kernel: conv2d_fp64(np.pad(image, (kernel.shape[0]//2, kernel.shape[1]//2), 'reflect'), kernel)
        conv_func = lambda image: pad_n_conv(image, _kernelic)
        gramian_func = lambda image: pad_n_conv(pad_n_conv(image, _kernelic), _kernelic_rev)

    elif mode == 'sparse':
        ...
    else:
        raise ValueError(f'"{mode}" unrecognized.')
    
    if with_gramian:
        return conv_func, gramian_func
    
    return conv_func


class ConvKernel2d:

    def __init__(self, kernel, mode='cv2', dual_kernel=None):

        assert mode in ['cv2', 'fft', 'sparse', 'scipy', 'dda4310whl']
        
        self.__kernel = kernel
        self.__filt, self.__gramian = make_conv_func(kernel, mode, with_gramian=True)

        if not isinstance(dual_kernel, ConvKernel2d):
            dual_kernel = ConvKernel2d(kernel[::-1,::-1], mode, self)
        
        self.__dual = dual_kernel

    def conv(self, image):
        return self.__filt(image)
    
    def convTconv(self, image):
        return self.__gramian(image)
    
    @property
    def T(self):
        return self.__dual
    
    @property
    def kernel(self):
        return self.__kernel
    
def gaussian_filt_kernel(shape, sigma):
    
    kernel = np.zeros(shape)
    mx, my = shape[0]//2, shape[1]//2
    dis2 = lambda x, y: (x-mx)**2 + (y-my)**2

    for i in range(shape[0]):
        for j in range(shape[1]):
            kernel[i, j] = np.exp(- dis2(i, j) / 2 / sigma**2)
    
    kernel /= np.sum(kernel)
    
    return kernel