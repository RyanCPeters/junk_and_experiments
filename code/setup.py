from setuptools import setup
from Cython.Build import cythonize

setup(name='Fractal period scanning',
      ext_modules=cythonize("fractal_tools.pyx"),
      zip_safe=False)
