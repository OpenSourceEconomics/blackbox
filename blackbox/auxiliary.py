"""This module hosts all auxiliary functions for running the BLACKBOX algorithm."""
import pickle as pkl
import datetime
import warnings
import shutil
import glob
import os

import scipy.optimize as op
import numpy as np

from blackbox.replacements_interface import get_capital_phi
from blackbox.replacements_interface import spread
from blackbox.executor_mpi import mpi_executor
from blackbox.executor_mp import mp_executor


def cubetobox_full(box, d, x):
    return [box[i][0]+(box[i][1]-box[i][0])*x[i] for i in range(d)]


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


class AggregatorCls(object):

    def __init__(self):
        """Constructor for the aggregation manager."""
        self.attr = dict()
        self.attr['fun_step'] = None
        self.attr['num_step'] = 0
        self.attr['num_eval'] = 0

    def run(self, e):
        """This function constantly checks for the best evaluation point."""
        fname = 'blackbox_best/smm_estimation/smm_monitoring.pkl'
        self.attr['fun_step'] = pkl.load(open(fname, 'rb'))['Current'].ix[0]

        # We need to set up the logging with the baseline information from the starting values.
        fmt_record = ' {:>25}{:>25.5f}{:>25.5f}{:>25}{:>25}\n'

        with open('blackbox.respy.log', 'w') as outfile:
            fmt_ = ' {:>25}{:>25}{:>25}{:>25}{:>25}\n\n'
            outfile.write(fmt_.format(*['Time', 'Criterion', 'Best', 'Evaluation', 'Step']))
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = list()
            line += [time, self.attr['fun_step'], self.attr['fun_step']]
            line += [self.attr['num_eval'], self.attr['num_step']]
            outfile.write(fmt_record.format(*line))

        self.attr['num_eval'] += 1

        # This setup ensures that a complete check of all directories is done after the algorithm
        # concludes. Otherwise, this might be terminated in an intermediate step.
        while not e.is_set():
            self._collect_information()
        else:
            self._collect_information()

    def _collect_information(self):
        """This method iterates over the BLACKBOX directories."""
        fmt_record = ' {:>25}{:>25.5f}{:>25.5f}{:>25}{:>25}\n'

        for dirname in glob.glob('blackbox_*'):
            if dirname in ['blackbox_best']:
                continue

            if os.path.exists(dirname + '/.ready.blackbox.info'):
                fname = dirname + '/smm_estimation/smm_monitoring.pkl'
                candidate = pkl.load(open(fname, 'rb'))['Current'].ix[0]
                if candidate < self.attr['fun_step']:
                    shutil.rmtree('blackbox_best')
                    shutil.copytree(dirname, 'blackbox_best')
                    self.attr['fun_step'] = candidate
                    self.attr['num_step'] += 1

                shutil.rmtree(dirname)

                # Update real-time logging
                with open('blackbox.respy.log', 'a') as outfile:
                    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    line = []
                    line += [time, candidate, self.attr['fun_step']]
                    line += [self.attr['num_eval'], self.attr['num_step']]
                    outfile.write(fmt_record.format(*line))
                self.attr['num_eval'] += 1


def fit_approx_model(batch, rho0, n, m, v1, fit, i, d, p, points):
    """This function fits the approximate model."""

    for j in range(batch):
        r = ((rho0 * ((m - 1. - (i * batch + j)) / (m - 1.)) ** p) / (
                v1 * (n + i * batch + j))) ** (1. / d)
        cons = [{'type': 'ineq', 'fun': lambda x, localk=k: np.linalg.norm(
            np.subtract(x, points[localk, 0:-1])) - r}
                for k in range(n + i * batch + j)]
        while True:
            minfit = op.minimize(fit, np.random.rand(d), method='SLSQP', bounds=[[0., 1.]] * d,
                                 constraints=cons)
            if not np.isnan(minfit.x)[0]:
                break
        points[n + i * batch + j, 0:-1] = np.copy(minfit.x)

    return points


def cleanup(is_start=False):
    """This module cleans the BLACKBOX directories."""
    for dirname in glob.glob('blackbox_*'):
        if not is_start and 'best' in dirname:
            continue
        shutil.rmtree(dirname)

    fname = 'blackbox.respy.csv'
    if is_start and os.path.exists(fname):
        os.remove(fname)

    fname = '.est_obj.blackbox.pkl'
    if os.path.exists(fname):
        os.remove(fname)


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
    d = len(points[0])-1

    Phi = get_capital_phi(points[:, 0:-1], T, n, d)

    P = np.ones((n, d+1))
    P[:, 0:-1] = points[:, 0:-1]

    F = points[:, -1]

    M = np.zeros((n+d+1, n+d+1))
    M[0:n, 0:n] = Phi
    M[0:n, n:n+d+1] = P
    M[n:n+d+1, 0:n] = np.transpose(P)

    v = np.zeros(n+d+1)
    v[0:n] = F

    try:
        sol = np.linalg.solve(M, v)
    except np.linalg.linalg.LinAlgError:
        warnings.warn('stabilization of BLACKBOX to avoid singular matrix')
        sol = np.ones(n+d+1)
    lam, b, a = sol[0:n], sol[n:n+d], sol[n+d]

    return lam, b, a


def get_executor(strategy, num_free=None, batch=None, crit_func=None):
    if strategy == 'mpi':
        # TODO: This needs to be cleanup by someone.
        pkl.dump(crit_func, open('.crit_func.blackbox.pkl', 'wb'))
        executor = mpi_executor(batch, num_free)
    elif strategy == 'mp':
        executor = mp_executor()
    else:
        raise NotImplementedError

    return executor


def bb_finalize(points, executor, strategy, fmax, cubetobox):
    # saving results into text file
    points[:, 0:-1] = list(map(cubetobox, points[:, 0:-1]))
    points[:, -1] = points[:, -1]*fmax
    points = points[points[:, -1].argsort()]

    if strategy == 'mpi':
        executor.terminate()

    return points
