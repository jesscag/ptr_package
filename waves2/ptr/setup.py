import distutils.core
import Cython.Build
import numpy as np

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
##python setup.py build_ext --inplace
'''
builds my pyx file to run test .pyx and includes numpy 
'''
distutils.core.setup(
    ext_modules=Cython.Build.cythonize("ptr.pyx"),
    include_dirs=[np.get_include()])

