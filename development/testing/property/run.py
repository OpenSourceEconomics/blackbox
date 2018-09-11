from scipy.optimize import rosen
import numpy as np

from blackbox.algorithm import search as pei_search
from ov_tools.tests.material.blackbox_original import search as bb_search
from blackbox.auxiliary import get_executor

open('.blackbox.testing.python', 'a').close()

for _ in range(10):
    d = np.random.randint(2, 5)
    box = [[-10., 10.]] * d

    n = d + np.random.randint(5, 10)
    m = np.random.randint(5, 25)
    batch = np.random.randint(1, 5)

    np.random.seed(123)
    alg_original = bb_search(rosen, box, n, m, batch, resfile='output.csv')  # text file where

    executor = get_executor('mp')

    np.random.seed(123)
    alg_revised = pei_search(rosen, box, n, m, batch, executor, 'mp', 'blackbox.respy.csv')

    np.testing.assert_almost_equal(alg_original, alg_revised)
