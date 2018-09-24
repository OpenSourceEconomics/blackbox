from functools import partial
import pickle as pkl

import numpy as np
import pyDOE

from blackbox.replacements_interface import fit_full
from blackbox.auxiliary import fit_approx_model
from blackbox.auxiliary import cubetobox_full
from blackbox.auxiliary import evaluate_batch
from blackbox.auxiliary import get_executor
from blackbox.auxiliary import bb_finalize
from blackbox.auxiliary import latin
from blackbox.auxiliary import rbf


def search(crit_func, box, n, m, batch, strategy, seed=123, legacy=False, rho0=0.5, p=1.0,
           nrand=10000, nrand_frac=0.05, is_restart=False):
    """
    Minimize given expensive black-box function and save results into text file.

    Parameters
    ----------
    crit_func : callable
        The objective function to be minimized.
    box : list of lists
        List of ranges for each parameter.
    n : int
        Number of initial function calls.
    m : int
        Number of subsequent function calls.
    batch : int
        Number of function calls evaluated simultaneously (in parallel).
    resfile : str
        Text file to save results.
    rho0 : float, optional
        Initial "balls density".
    p : float, optional
        Rate of "balls density" decay (p=1 - linear, p>1 - faster, 0<p<1 - slower).
    nrand : int, optional
        Number of random samples that is generated for space rescaling.
    nrand_frac : float, optional
        Fraction of nrand that is actually used for space rescaling.
    executor : callable, optional
        Should have a map method and behave as a context manager.
        Allows the user to use various parallelisation tools
        as dask.distributed or pathos.
    """
    # TODO: We need a setup where we check all input parameters of the request that no subsequent
    # termination is possible.
    import os
    if os.path.exists("fitting.blackbox.log"):
        os.remove("fitting.blackbox.log")

    if seed is not None:
        np.random.seed(seed)

    # We distinguish parallelism between MPI and MP implementations.
    # space size
    d = len(box)

    # adjusting the number of function calls to the batch size
    if n % batch != 0:
        n = n - n % batch + batch

    if m % batch != 0:
        m = m - m % batch + batch

    # TODO: This does possibly start more procs then requested, this should be revisited.
    executor = get_executor(strategy, d, batch, crit_func)

    # We are ready to define the auxiliary functions.
    cubetobox = partial(cubetobox_full, box, d)

    if not is_restart:

        # generating latin hypercube
        points = np.zeros((n, d + 1))
        if not legacy:
            points[:, 0:-1] = pyDOE.lhs(d, samples=n)
        else:
            points[:, 0:-1] = latin(n, d)

        # initial sampling
        for i in range(n // batch):
            candidates = list(map(cubetobox, points[batch * i:batch * (i + 1), 0:-1]))
            stat = evaluate_batch(strategy, executor, crit_func, candidates)
            points[batch * i:batch * (i + 1), -1] = stat

        # normalizing function values
        fmax = max(abs(points[:, -1]))

        if fmax == 0.0:
            raise NotImplementedError

        points[:, -1] = points[:, -1] / fmax

        # Prepare information for potential restart.
        restart_material = dict()
        restart_material['points'] = points
        restart_material['fmax'] = fmax

        pkl.dump(restart_material, open('restart.blackbox.pkl', 'wb'))

    else:

        # TODO: We need to add tests that this is a proper restart by at least checking the
        # number of free parameters, not nan, etc.
        restart_material = pkl.load(open('restart.blackbox.pkl', 'rb'))
        points, fmax = restart_material['points'], restart_material['fmax']

    # This allows to request a simple search on a random grid.
    if m == 0:
        return bb_finalize(points, executor, strategy, fmax, cubetobox)

    # volume of d-dimensional ball (r = 1)
    if d % 2 == 0:
        v1 = np.pi ** (d / 2) / np.math.factorial(d / 2)
    else:
        v1 = 2 * (4 * np.pi) ** ((d - 1) / 2) * np.math.factorial((d - 1) / 2)
        v1 /= np.math.factorial(d)

    # subsequent iterations (current subsequent iteration = i*batch+j)
    T = np.identity(d)

    for i in range(m // batch):

        # refining scaling matrix T
        if d > 1:
            lam, b, a = rbf(points, np.identity(d))
            fit_noscale = partial(fit_full, lam, b, a, np.identity(d), points[:, 0:-1])

            population = np.zeros((nrand, d + 1))
            population[:, 0:-1] = np.random.rand(nrand, d)
            population[:, -1] = list(map(fit_noscale, population[:, 0:-1]))

            cloud = population[population[:, -1].argsort()][0:int(nrand * nrand_frac), 0:-1]
            eigval, eigvec = np.linalg.eig(np.cov(np.transpose(cloud)))
            T = [eigvec[:, j] / np.sqrt(eigval[j]) for j in range(d)]
            T = T / np.linalg.norm(T)

        # sampling next batch of points
        lam, b, a = rbf(points, T)
        fit = partial(fit_full, lam, b, a, T, points[:, 0:-1])

        # TODO: THe approximation depends on the batch size, Maybe it is a good idea to split
        # number of points for approximation and batch size by using a queue instead.
        # TODO: This is scalar at this point, other cores could evaluate other random points in
        # the meantime?
        points = np.append(points, np.zeros((batch, d + 1)), axis=0)
        points = fit_approx_model(batch, rho0, n, m, v1, fit, i, d, p, points)

        candidates = list(map(cubetobox, points[n + batch * i:n + batch * (i + 1), 0:-1]))
        stat = evaluate_batch(strategy, executor, crit_func, candidates)
        points[n + batch * i:n + batch * (i + 1), -1] = stat / fmax

    points = bb_finalize(points, executor, strategy, fmax, cubetobox)

    return points
