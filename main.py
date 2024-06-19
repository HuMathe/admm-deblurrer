import os
import cv2
import argparse
import numpy as np

from admm.admm_deblurrer import ImageDeblurrer as AdmmDeblurSolver
from utils.patchify_utils import pathified_map
from utils.conv_utils import gaussian_filt_kernel as gkernel

class ImageDeblurrer:
    
    def __init__(self, args):

        self.args = args
        self.patch_size = args.patch_height, args.patch_width
        self.pad_size = args.gaus_size // 2, args.gaus_size//2

        self.build_solver_kwargs = {
            'lamd': args.lamd,
            'rho': args.rho,
            'blur_filter': gkernel((args.gaus_size, args.gaus_size), args.gaus_sigma),
            'mode': args.admm_mode,
        }
        self.solver_cache = {}

    def deblur_map(self, image):

        solver = self.get_solver(image.shape)
        ret_image = solver.solve_inverse_problem(image, 50)

        return ret_image
    
    def get_solver(self, input_shape):

        if input_shape in self.solver_cache:
            return self.solver_cache[input_shape]
        
        solver = AdmmDeblurSolver(input_shape, **self.build_solver_kwargs)
        
        self.solver_cache[input_shape] = solver
        
        return solver

    def deblur_single_channel(self, image):
        # DEBUG

        # return self.deblur_map(image[200:200+self.patch_size[0],400:400+self.patch_size[1]])

        return pathified_map(
            self.deblur_map, 
            image, 
            self.pad_size,
            self.patch_size,
        )

def main(args):
    
    if not os.path.exists(args.image):
        print(f'Image path: {args.image} does not exist!')
        exit(1)
    
    deblurrer = ImageDeblurrer(args)

    raw_image = cv2.imread(args.image)
    
    bgr = cv2.split(raw_image / 255)
    res_image = cv2.merge(list(map(deblurrer.deblur_single_channel, bgr)))
    res_image = np.clip(res_image * 255, 0, 255).astype(np.uint8)

    # save image
    print(f'result saving to {args.save_path}')
    cv2.imwrite(args.save_path, res_image)


parser = argparse.ArgumentParser(description='Image Deblurring Options')

parser.add_argument('-i', '--image', help='image load path', required=True)
parser.add_argument('-s', '--save_path', help='image save path', default='result.png')
parser.add_argument('-r', '--rho', help='admm solver para: rho', default=0.1, type=float)
parser.add_argument('-l', '--lamd', help='reg. para: lambda', default=0.003, type=float)
parser.add_argument('-ph', '--patch_height', help='image patch height', default=256, type=int)
parser.add_argument('-pw', '--patch_width', help='image path width', default=256, type=int)
parser.add_argument('-gs', '--gaus_size', help='gaussian blurring size', default=17, type=int)
parser.add_argument('-gstd', '--gaus_sigma', help='gaussian blurring sigma', default=5, type=float)
parser.add_argument('--admm_mode', help='admm convolution implementation', choices=['cv2', 'sparse', 'fft', 'dda4310'], default='cv2')


if __name__ == '__main__':

    args = parser.parse_args()

    main(args)