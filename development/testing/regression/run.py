#!/usr/bin/env python
"""This script allows to create and run the regression test."""
import pickle as pkl
import os

from scipy.optimize import rosen
import numpy as np

from blackbox.tests.auxiliary import get_valid_request
from blackbox import search
import blackbox


is_creation = False

if is_creation:
    num_tests = 100
    tests = []
    for seed in range(num_tests):

        request = [rosen] + list(get_valid_request())[1:]
        rslt = search(*request)
        tests.append([request, rslt])
    pkl.dump(tests, open('regression_vault.blackbox.pkl', 'wb'))

# TODO: These are only working on heracles at the moment, portability is still an issue.
FNAME_VAULT = os.path.dirname(blackbox.__file__) + '/tests/material/regression_vault.blackbox.pkl'
for i, test in enumerate(pkl.load(open(FNAME_VAULT, 'rb'))):
    print(" running test " + str(i))
    request, rslt = test
    stat = search(*request)

    np.testing.assert_equal(rslt, stat)
