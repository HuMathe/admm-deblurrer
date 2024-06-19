import numpy as np
from functools import lru_cache

def convFilter(image, kernel):
    
    KH, KW = kernel.shape
    PH, PW = KH//2, KW//2

    padded = np.pad(image, (PH, PW), 'reflect')
    fft_a = np.fft.fft2(padded)
    fft_b = np.fft.fft2(kernel, s=padded.shape)
    
    result = np.fft.ifft2(fft_a * fft_b).real[PH*2:, PW*2:]
    
    return result

def convGramian(image, kernel):

    KH, KW = kernel.shape
    PH, PW = KH // 2, KW // 2

    padded = np.pad(image, (PH*2, PW*2), 'reflect')
    fft_a = np.fft.fft2(padded)
    fft_b = np.fft.fft2(kernel, s=padded.shape)
    fft_c = np.fft.fft2(kernel[::-1,::-1], s=padded.shape)

    result = np.fft.ifft2(fft_a * fft_b * fft_c).real[PH*4:,PW*4:]

    return result

def convFilterFunc(kernel):

    kernel = np.copy(kernel)
    KH, KW = kernel.shape
    PH, PW = KH//2, KW//2

    @lru_cache(None)
    def __get_kernel_fft_feature(padded_shape):

        fft_featue = np.fft.fft2(kernel, s=padded_shape)

        return fft_featue
    
    def oncall(image):
        
        padded = np.pad(image, (PH, PW), 'reflect')

        fft_a = np.fft.fft2(padded)
        fft_b = __get_kernel_fft_feature(padded.shape)
    
        result = np.fft.ifft2(fft_a * fft_b).real[PH*2:, PW*2:]
    
        return result
    
    return oncall

def convGramianFunc(kernel):

    kernel = np.copy(kernel)
    KH, KW = kernel.shape
    PH, PW = KH // 2, KW // 2

    @lru_cache(None)
    def __get_kernel_fft_feature(padded_shape):

        fft_featue = np.fft.fft2(kernel, s=padded_shape)
        fft_rev_featue = np.fft.fft2(kernel[::-1,::-1], s=padded_shape)
        ret = fft_featue * fft_rev_featue

        return ret

    def oncall(image):

        padded = np.pad(image, (PH*2, PW*2), 'reflect')

        fft_a = np.fft.fft2(padded)
        fft_b = __get_kernel_fft_feature(padded.shape)

        result = np.fft.ifft2(fft_a * fft_b).real[PH*4:,PW*4:]
        
        return result
    
    return oncall