import pickle as pkl

from scipy.optimize import rosen
import numpy as np

from blackbox import search
from blackbox.tests.auxiliary import get_valid_request
FNAME_VAULT = 'regression_vault.blackbox.pkl'


num_tests = 100

tests = []
for seed in range(num_tests):

    request = [rosen] + list(get_valid_request())[1:]
    rslt = search(*request, seed=123)
    tests.append([request, rslt])

pkl.dump(tests, open(FNAME_VAULT, 'wb'))


for test in pkl.load(open(FNAME_VAULT, 'rb')):
    request, rslt = test
    stat = search(*request, seed=123)

    np.testing.assert_equal(rslt, stat)
