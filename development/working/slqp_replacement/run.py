import pickle as pkl
from functools import partial
import numpy as np
from blackbox.replacements_interface import constraint_full, fit_full
from scipy.optimize import minimize
from blackbox.auxiliary import rbf
import datetime
#
# restart_dict = pkl.load(open('restart.blackbox.pkl', 'rb'))
#
# points = restart_dict['points']
#
# n = points.shape[0]
# r = 0.3
# d = points.shape[1] - 1
#
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
#
#
# # TODO: The original code allows for repeated attempts for the SLSQP routine.
# # However, this was not ever required in production.
# start = np.random.rand(d)
#
# now = datetime.datetime.now()
# print('start ', now.strftime("%H:%M:%S"))
# rslt = minimize(fit, start, method='SLSQP', bounds=bounds, constraints=cons)
# now = datetime.datetime.now()
# print(rslt)
# print('end ', now.strftime("%H:%M:%S"))


from replacements_f2py import minimize_slsqp
minimize_slsqp(np.random.uniform(size=5), 5)
