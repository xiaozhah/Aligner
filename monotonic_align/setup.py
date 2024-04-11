from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)

# cd monotonic_align; mkdir monotonic_align; python setup.py build_ext --inplace