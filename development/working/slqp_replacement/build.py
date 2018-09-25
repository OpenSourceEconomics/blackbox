import os

from numpy import f2py

cmd = 'gfortran -c -fpic shared_constants.f90 slsqp_interface.f90 slsqp.f'
os.system(cmd)

cmd = 'ar cr libblackbox.a *.o'
os.system(cmd)

args = '-I. -L. -lblackbox '
src = open('replacements_f2py.f90', 'rb').read()
f2py.compile(src, 'replacements_f2py', args, extension='.f90')
