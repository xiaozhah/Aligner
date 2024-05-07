from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'robo_utils',
    ext_modules=cythonize("core.pyx"),
    include_dirs=[numpy.get_include()]
)

# cd robo_utils; mkdir robo_utils; python setup.py build_ext --inplace