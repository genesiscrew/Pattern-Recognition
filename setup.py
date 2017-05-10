from distutils.core import setup
from Cython.Build import cythonize

setup(name="OurCode5", ext_modules=cythonize('OurCode5.pyx'),)