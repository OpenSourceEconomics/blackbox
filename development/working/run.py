from blackbox import search
from scipy.optimize import rosen
import cProfile
import pstats

d = 2
box = [[-10, 10]] * d
n = 100
m = 10
strategy = 'mp'
batch = 2

search(rosen, box, n, m, batch, strategy)
