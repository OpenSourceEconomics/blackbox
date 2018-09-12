import os

from numpy import f2py

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


def compile_f2py():
    """This function compiles the F2PY interface."""
    base_path = os.getcwd()
    os.chdir(PACKAGE_DIR)

    args = '--f90flags="-ffree-line-length-0 -O3"'
    src = open('replacements_f2py.f90', 'rb').read()
    f2py.compile(src, 'replacements_f2py', args, extension='.f90')

    os.chdir(base_path)


try:
    import blackbox.replacements_f2py  # noqa: F401
except ModuleNotFoundError:
    compile_f2py()

from blackbox.algorithm import search  # noqa: F401
