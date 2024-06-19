from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

__version__ = '0.1.0'

# Define the extension module
extension_mod = Extension(
    'simple_conv2d',
    sources=['simple-conv.cpp'],
    language='c++',
    include_dirs=['pybind11/include'],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)

# Custom build extension command to enable C++11 support
class BuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler
        if c.compiler_type == 'unix':
            if sys.platform == 'darwin':
                self.extensions[0].extra_compile_args.append('-stdlib=libc++')
                self.extensions[0].extra_link_args.append('-stdlib=libc++')
            else:
                self.extensions[0].extra_compile_args.append('-std=c++11')
        elif c.compiler_type == 'msvc':
            self.extensions[0].extra_compile_args.append('/std:c++11')

        build_ext.build_extensions(self)

# Setup configuration
setup(
    name='simple_conv2d',
    version=__version__,
    author='Derong Jin',
    author_email='derongjin19@gmail.com',
    description='A C++ module for 2d convolution',
    ext_modules=[extension_mod],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False
)