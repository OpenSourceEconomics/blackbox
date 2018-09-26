import os

from numpy import f2py

FLAGS_DEBUG = []
FLAGS_DEBUG += ['-O', '-Wall', '-Wline-truncation', '-Wsurprising', '-Waliasing']
FLAGS_DEBUG += ['-Wunused-parameter', '-fwhole-file', '-fcheck=all']
FLAGS_DEBUG += ['-fbacktrace', '-g', '-fmax-errors=1', '-ffree-line-length-0']
FLAGS_DEBUG += ['-cpp', '-Wcharacter-truncation', '-Wimplicit-interface']

FLAGS_PRODUCTION = ['-O3', '-ffree-line-length-0']

args = ' '.join(FLAGS_PRODUCTION)

os.chdir('src')

cmd = ''
cmd += 'gfortran -c ' + args + ' -fpic shared_constants.f90 slsqp_interface.f90 slsqp.f '
cmd += ' replacements.f90 blackbox.f90'
os.system(cmd)

cmd = 'ar cr libblackbox.a *.o'
os.system(cmd)
os.chdir('../')

args = '--f90flags="' + args + '" -Isrc -Lsrc -lblackbox '
src = open('replacements_f2py.f90', 'rb').read()
f2py.compile(src, 'replacements_f2py', args, extension='.f90')
