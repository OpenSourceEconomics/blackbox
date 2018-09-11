
from blackbox.replacements_interface import spread
from blackbox.algorithm import latin
from blackbox.auxiliary import rbf

import numpy as np

d = 20
n = 500

points = np.zeros((n, d + 1))
points[:, 0:-1] = latin(n, d)

mat, eval_points = np.identity(d), np.random.rand(d)
print("gonig in ")
lam, b, a = rbf(points, mat)
print("gonig out ")
spread(points[:, 0:-1], n, d)
