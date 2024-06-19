
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from .admm_base import AdmmBase
from utils import (
    conv_utils,
)

def Sk(v, k):
    max_0 = lambda x: np.clip(x, 0, x.max())
    return max_0(v-k) - max_0(-v-k)

class ImageDeblurrer(AdmmBase):
    
    def __init__(self, input_shape, lamd, rho, blur_filter, mode='cv2'):
        super().__init__(input_shape)
        self.lamd = lamd
        self.rho = rho
        self.bker = conv_utils.ConvKernel2d(blur_filter, mode)
        self.Dx = conv_utils.ConvKernel2d(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]), mode)
        self.Dy = conv_utils.ConvKernel2d(np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]), mode)
        
        self.hessian_op = lambda x: self.bker.convTconv(x) \
            + self.rho * (self.Dx.convTconv(x) + self.Dy.convTconv(x))

        self.HLrho = LinearOperator(
            (np.prod(input_shape), np.prod(input_shape)),
            matvec = (lambda x: self.hessian_op(x.reshape(input_shape)).ravel()),
            rmatvec = (lambda x: self.hessian_op(x.reshape(input_shape)).ravel()))
    
    def admm_init_xzub(self, b):
        x, zx, zy, ux, uy = [np.zeros(self.shape) for _ in range(5)]
        return x, (zx, zy), (ux, uy), b
    
    def admm_x_step(self, x, z, u, b, cg_max_iter=100, **kwargs):
        zx, zy = z
        ux, uy = u
        b_ = self.bker.T.conv(b) \
            - self.rho * (self.Dx.T.conv(ux - zx) + self.Dy.T.conv(uy - zy))
        x_, _ = cg(self.HLrho, b_.flatten(), x.flatten(), maxiter=cg_max_iter)
        return x_.reshape(self.shape)
    
    def admm_z_step(self, x, z, u, b, **kwagrs):
        ux, uy = u
        zx = Sk(self.Dx.conv(x)+ ux, self.lamd / self.rho)
        zy = Sk(self.Dy.conv(x)+ uy, self.lamd / self.rho)
        return (zx, zy)
    
    def admm_u_step(self, x, z, u, b, **kwargs):
        ux, uy = u
        zx, zy = z
        ux = ux + self.Dx.conv(x) - zx
        uy = uy + self.Dy.conv(x) - zy
        return (ux, uy)

    def admm_refine_x_step(self, x, z, u, b, **kwargs):
        return self.admm_x_step(x, z, u, b, None)


