import pickle as pkl

from scipy.optimize import rosen
import numpy as np

from blackbox.algorithm import search as pei_search
from blackbox.auxiliary import get_executor

FNAME_VAULT = 'regression_vault.blackbox.pkl'


def get_request(seed):

    np.random.seed(seed)

    d = np.random.randint(2, 5)
    box = [[-10., 10.]] * d

    n = d + np.random.randint(0, 5)
    m = np.random.randint(2, 10)
    batch = np.random.randint(2, 5)
    executor = get_executor('mp')

    return box, n, m, batch, executor
#
# tests = []
# for seed in range(100):
#
#     box, n, m, batch, executor = get_request(seed)
#     rslt = pei_search(rosen, box, n, m, batch, executor, 'mp', 'blackbox.respy.csv')
#
#     tests.append([seed, rslt])
#
# pkl.dump(tests, open(FNAME_VAULT, 'wb'))


for test in pkl.load(open(FNAME_VAULT, 'rb')):
    seed, rslt = test

    box, n, m, batch, executor = get_request(seed)
    stat = pei_search(rosen, box, n, m, batch, executor, 'mp')

    np.testing.assert_equal(rslt, stat)
