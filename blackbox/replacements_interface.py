"""This module serves as a common interface for selected functions of the algorithm that are
available in FORTRAN and PYTHON. It allows to easily switch between the implementations to ease
testing."""
import os

from blackbox.replacements_pyth import pyth_get_capital_phi
from blackbox.replacements_pyth import pyth_fit_full
from blackbox.replacements_pyth import pyth_spread

from blackbox.replacements_f2py import f2py_get_capital_phi
from blackbox.replacements_f2py import f2py_fit_full
from blackbox.replacements_f2py import f2py_spread


def spread(points, n, d):
    """This is the interface to the different implementations of the spread()."""
    if os.path.exists('.blackbox.testing.python'):
        rslt = pyth_spread(points, n, d)
    else:
        rslt = f2py_spread(points, n, d)
    return rslt


def fit_full(lam, b, a, T, points, x):
    """This is the interface to the different """
    if os.path.exists('.blackbox.testing.python'):
        rslt = pyth_fit_full(lam, b, a, T, points, x)
    else:
        rslt = f2py_fit_full(lam, b, a, T, points, x)
    return rslt


def get_capital_phi(points, T, n, d):
    """This is the interface to the different implementations of the get_capital_phi()."""
    if os.path.exists('.blackbox.testing.python'):
        rslt = pyth_get_capital_phi(points, T, n, d)
    else:
        rslt = f2py_get_capital_phi(points, T, n, d)
    return rslt
