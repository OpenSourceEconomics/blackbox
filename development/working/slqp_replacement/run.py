import pickle as pkl
from functools import partial
import numpy as np
from scipy.optimize import minimize
from blackbox.auxiliary import rbf
import datetime
import blackbox.replacements_f2py as f2py


from blackbox.tests.material.blackbox_original import search as bb_search
from blackbox.replacements_interface import get_capital_phi
from blackbox.replacements_interface import constraint_full
from blackbox.tests.auxiliary import get_valid_request
from blackbox.replacements_interface import fit_full
from blackbox.replacements_interface import spread
from blackbox.tests.auxiliary import EXECUTORS
from blackbox.algorithm import latin
from blackbox.auxiliary import rbf
from blackbox import PACKAGE_DIR
from blackbox import search
import datetime
restart_dict = pkl.load(open('restart.blackbox.pkl', 'rb'))
points = restart_dict['points']




for _ in range(10):
    print(_)
    d, _, n, _, _, _ = get_valid_request()
    points = np.zeros((n, d + 1))
    points[:, 0:-1] = latin(n, d)

    #d = points.shape[1] - 1
    #n = points.shape[0]

    r = np.random.uniform() + 0.01
    mat, x = np.identity(d), np.random.rand(d)
    lam, b, a = rbf(points, mat)
    T = mat
    print('going in ')
    x = f2py.f2py_minimize_slsqp(x, r, points[:,0:-1], lam, b, a, T, int(d), n)
    print('going out ')

#
# points = points[:100, :]
#
# n = points.shape[0]
# r = 0.3
# d = points.shape[1] - 1
# #
# mat = np.identity(d)
# T = mat
# lam, b, a = rbf(points, mat)
# k = np.random.choice(range(n))
# r = np.random.uniform()
# x = np.random.uniform(size=d)
#
# cons = list()
# for k in range(n):
#     constraint = partial(constraint_full, points, r, k)
#     cons.append({'type': 'ineq', 'fun': constraint})
#
# bounds = [[0.0, 1.0]] * d
# fit = partial(fit_full, lam, b, a, T, points[:, 0:-1])
# #
# #
# # # TODO: The original code allows for repeated attempts for the SLSQP routine.
# # # However, this was not ever required in production.
# start = np.random.rand(d)
# #
# now = datetime.datetime.now()
# print('start ', now.strftime("%H:%M:%S"))
# rslt = minimize(fit, start, method='SLSQP', bounds=bounds, constraints=cons)
# x = f2py.f2py_minimize_slsqp(start, r, points[:,0:-1], lam, b, a, T, int(d), n)
#
# now = datetime.datetime.now()
# print(rslt['x'], x)
# print('end ', now.strftime("%H:%M:%S"))
