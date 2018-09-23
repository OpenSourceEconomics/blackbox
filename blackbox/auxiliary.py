"""This module hosts all auxiliary functions for running the BLACKBOX algorithm."""
from functools import partial
import multiprocessing as mp
import pickle as pkl
import warnings

import scipy.optimize as op
import numpy as np

from blackbox.replacements_interface import get_capital_phi
from blackbox.replacements_interface import spread
from blackbox.executor_mpi import mpi_executor


def evaluate_batch(strategy, executor, crit_func, candidates):
    """This function evaluates the batch with the available executor."""
    if strategy == 'mpi':
        stat = executor.evaluate(candidates)
    else:
        with executor() as e:
            stat = list(e.map(crit_func, candidates))
    return stat


def cubetobox_full(box, d, x):
    """This function transfers the points back to their original sizes."""
    rslt = list()
    for i in range(d):
        rslt.append(box[i][0] + (box[i][1] - box[i][0]) * x[i])
    return rslt


def latin(n, d):
    """
    Build latin hypercube.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Size of space.

    Returns
    -------
    lh : ndarray
        Array of points uniformly placed in d-dimensional unit cube.
    """
    # spread function

    # starting with diagonal shape
    lh = [[i/(n-1.)]*d for i in range(n)]

    # minimizing spread function by shuffling
    minspread = spread(lh, n, d)

    for i in range(1000):
        point1 = np.random.randint(n)
        point2 = np.random.randint(n)
        dim = np.random.randint(d)

        newlh = np.copy(lh)
        newlh[point1, dim], newlh[point2, dim] = newlh[point2, dim], newlh[point1, dim]
        newspread = spread(newlh, n, d)

        if newspread < minspread:
            lh = np.copy(newlh)
            minspread = newspread

    return lh


def fit_approx_model(batch, rho0, n, m, v1, fit, i, d, p, points):
    """This function fits the approximate model."""
    def constraint_full(k, r, x):
        return np.linalg.norm(np.subtract(x, points[k, 0:-1])) - r

    # We try to learn more about the performance problems.
    fname = 'fitting.blackbox.log'
    import os

    if not os.path.exists(fname):
        os.mknod(fname)
    import datetime

    with open(fname, 'a') as outfile:

        now = datetime.datetime.now()
        outfile.write('\n\n Starting on new badge ' + now.strftime("%H:%M:%S") + '\n')

        for j in range(batch):

            now = datetime.datetime.now()
            outfile.write('    Starting on new point ' + now.strftime("%H:%M:%S") + '\n')

            r = ((rho0 * ((m - 1. - (i * batch + j)) / (m - 1.)) ** p) / (v1 * (n + i * batch + j)))
            r **= (1. / d)

            # We need to construct a full set of bounds.
            # TODO: NOte that the bounds are a direct function of the explorative function calls n.
            cons = list()
            for k in range(n + i * batch + j):
                constraint = partial(constraint_full, k, r)
                cons.append({'type': 'ineq', 'fun': constraint})

            bounds = [[0.0, 1.0]] * d

            count = 1
            while True:
                start = np.random.rand(d)
                rslt_x = op.minimize(fit, start, method='SLSQP', bounds=bounds, constraints=cons).x
                if not np.isnan(rslt_x)[0]:
                    break
                count += 1

            now = datetime.datetime.now()
            outfile.write('    Finished on new point ' + now.strftime("%H:%M:%S") + ' after ' +
                          str(count) + ' attempts \n\n')

            points[n + i * batch + j, 0:-1] = np.copy(rslt_x)

    return points


def rbf(points, T):
    """
    Build RBF-fit for given points (see Holmstrom, 2008 for details) using scaling matrix.

    Parameters
    ----------
    points : ndarray
        Array of multi-d points with corresponding values [[x1, x2, .., xd, val], ...].
    T : ndarray
        Scaling matrix.

    Returns
    -------
    fit : callable
        Function that returns the value of the RBF-fit at a given point.
    """
    n = len(points)
    d = len(points[0]) - 1

    Phi = get_capital_phi(points[:, 0:-1], T, n, d)

    P = np.ones((n, d + 1))
    P[:, 0:-1] = points[:, 0:-1]

    F = points[:, -1]

    M = np.zeros((n + d + 1, n + d + 1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n + d + 1] = P
    M[n:n + d + 1, 0:n] = np.transpose(P)

    v = np.zeros(n + d + 1)
    v[0:n] = F

    try:
        sol = np.linalg.solve(M, v)
    except np.linalg.linalg.LinAlgError:
        warnings.warn('stabilization of BLACKBOX to avoid singular matrix')
        sol = np.ones(n + d + 1)
    lam, b, a = sol[0:n], sol[n:n + d], sol[n + d]

    return lam, b, a


def get_executor(strategy, num_free=None, batch=None, crit_func=None):
    """This function returns the executor for the evaluation of points."""
    if strategy == 'mpi':
        pkl.dump(crit_func, open('.crit_func.blackbox.pkl', 'wb'))
        executor = mpi_executor(batch, num_free)
    elif strategy == 'mp':
        executor = mp.Pool
    else:
        raise NotImplementedError

    return executor


def bb_finalize(points, executor, strategy, fmax, cubetobox):
    """This functions finalizes the BLACKBOX algorithm."""
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1]*fmax
    points = points[points[:, -1].argsort()]

    if strategy == 'mpi':
        executor.terminate()

    return points
