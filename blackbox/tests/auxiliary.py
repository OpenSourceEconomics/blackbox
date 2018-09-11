import os
import numpy as np

# Check the set of available executors.
EXECUTORS = ['mp']
if 'PMI_SIZE' in os.environ.keys():
    EXECUTORS.append('mpi')


def get_valid_request():
    """This function generates a valid request for the BLACKBOX algorithm."""

    d = np.random.randint(2, 5)
    box = [[-10., 10.]] * d

    n = d + np.random.randint(5, 10)
    m = np.random.randint(5, 25)
    batch = np.random.randint(1, 5)

    strategy = np.random.choice(EXECUTORS)

    return d, box, n, m, batch, strategy
