import numpy as np


def phi(r):
    return r * r * r


def pyth_spread(points, n, d):

        return sum(1./np.linalg.norm(np.subtract(points[i], points[j])) for i in range(n)
                   for j in range(n) if i > j)


def pyth_fit_full(lam, b, a, T, points, x):
    n = len(points)
    return sum(lam[i] * phi(np.linalg.norm(np.dot(T, np.subtract(x, points[i, :])))) for i in
               range(n)) + np.dot(b, x) + a


def pyth_get_capital_phi(points, T, n, d):
    return [[phi(np.linalg.norm(np.dot(T, np.subtract(points[i, :], points[j, :])))) for j in
             range(n)] for i in range(n)]


def pyth_constraint_full(point, r, x):
    return np.linalg.norm(np.subtract(x, point)) - r
