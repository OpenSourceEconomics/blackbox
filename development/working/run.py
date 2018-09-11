from blackbox import search
from scipy.optimize import rosen
import cProfile
import pstats
d = 2
box = [[-10, 10]] * d
n = 100
m = 10
strategy = 'mp'
batch = 20

cProfile.run('search(rosen, box, n, m, batch, strategy)', 'test.prof')

p = pstats.Stats("test.prof")
p.strip_dirs().sort_stats('cumulative').print_stats()