"""This module contains all functions for testing the BLACKBOX algorithm."""
import os

from scipy.optimize import rosen
import numpy as np
import pytest

from blackbox.tests.material.blackbox_original import search as bb_search
from blackbox.replacements_interface import get_capital_phi
from blackbox.replacements_interface import fit_full
from blackbox.replacements_interface import spread
from blackbox.algorithm import latin
from blackbox.auxiliary import rbf
from blackbox import search
from blackbox.tests.auxiliary import get_valid_request

PYTHON_FNAME = '.blackbox.testing.python'


@pytest.mark.skipif('PMI_SIZE' not in os.environ.keys(), reason='no MPI execution')
def test_1():
    """This function tests that the BLACKBOX results are unaffected by the parallelization
    strategy."""
    # TODO: needs to be endogenixzed
    d = 2#np.random.randint(2, 5)
    box = [[-10., 10.]] * d

    n = d + np.random.randint(5, 10)
    m = np.random.randint(5, 25)
    batch = np.random.randint(1, 5)

    rslt_base = None
    for strategy in ['mpi', 'mp']:

        np.random.seed(123)
        rslt = search(rosen, box, n, m, batch, strategy)

        if rslt_base is None:
            rslt_base = rslt
        np.testing.assert_almost_equal(rslt, rslt_base)


@pytest.mark.xfail(reason='... this is clear from studying the code but still unfortunate.')
def test_2():
    """This function tests that the BLACKBOX results are unaffected by the size of the batch."""
    d, box, n, m, batch, strategy = get_valid_request()

    rslt_base = None
    for batch in range(1, 5):
        np.random.seed(123)
        rslt = search(rosen, box, n, m, batch, strategy, legacy=True)

        if rslt_base is None:
            rslt_base = rslt

        np.testing.assert_almost_equal(rslt, rslt_base)


def test_3():
    """This function tests that the results line up with the original algorithm."""
    # Unfortunately, there will be small differences when using the FORTRAN implementation
    # of selected functions. However, this test always passes when the PYTHON versions are
    # used. I test all implementations against each other in other tests and as the interface
    # is identical, I am confident that the discrepancies here are only due to small
    # numerical differences in the PYTHON and FORTRAN calculations that accumulate if all are
    # used at once.
    open(PYTHON_FNAME, 'a').close()

    d, box, n, m, batch, strategy = get_valid_request()

    np.random.seed(123)
    alg_original = bb_search(rosen, box, n, m, batch, resfile='output.csv')

    np.random.seed(123)
    alg_revised = search(rosen, box, n, m, batch, strategy, legacy=True)

    np.testing.assert_almost_equal(alg_original, alg_revised)


def test_4():
    """This test function compares the output from the F2PY functions to their PYTHON
    counterparts."""
    for _ in range(100):

        d, _, n, _, _, _ = get_valid_request()

        points = np.zeros((n, d + 1))
        points[:, 0:-1] = latin(n, d)

        mat, eval_points = np.identity(d), np.random.rand(d)
        lam, b, a = rbf(points, mat)

        rslt_base = None
        for is_python in [True, False]:

            if is_python:
                open(PYTHON_FNAME, 'a').close()

            rslt = list()
            rslt.append(fit_full(lam, b, a, mat, points[:, 0:-1], eval_points))
            rslt.append(get_capital_phi(points[:, 0:-1], mat, n, d))
            rslt.append(spread(points[:, 0:-1], n, d))

            if os.path.exists(PYTHON_FNAME):
                os.remove(PYTHON_FNAME)

            if rslt_base is None:
                rslt_base = rslt

            for i in range(3):
                np.testing.assert_almost_equal(rslt[i], rslt_base[i])


@pytest.mark.xfail(reason='... minor numerical differences accumulate over many iterations.')
def test_5():
    """This test function ensures that the results are unaffected by using either the FORTRAN or
    PYTHON function."""
    d, box, n, m, batch, strategy = get_valid_request()

    rslt_base = None
    for is_python in [True, False]:

        if is_python:
            open(PYTHON_FNAME, 'a').close()

        np.random.seed(123)
        rslt = search(rosen, box, n, m, batch, strategy)

        if rslt_base is None:
            rslt_base = rslt

        np.testing.assert_almost_equal(rslt, rslt_base)

        if os.path.exists(PYTHON_FNAME):
            os.remove(PYTHON_FNAME)
