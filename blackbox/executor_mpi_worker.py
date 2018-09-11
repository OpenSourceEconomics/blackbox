#!/usr/bin/env python
import pickle as pkl
import numpy as np

from mpi4py import MPI


comm = MPI.Comm.Get_parent()
num_slaves, rank = comm.Get_size(), comm.Get_rank()

crit_func = pkl.load(open('.crit_func.blackbox.pkl', 'rb'))

num_free = np.array(0, dtype='int32')
comm.Bcast([num_free, MPI.INT], root=0)

while True:

    cmd = np.array(0, dtype='int32')
    comm.Bcast([cmd, MPI.INT], root=0)

    if cmd == -1:
        comm.Disconnect()
        break

    elif cmd == 1:

        x_free_econ_eval = np.tile(np.nan, num_free)
        sendbuf = np.tile(np.nan, (num_slaves, num_free))
        comm.Scatter(sendbuf, x_free_econ_eval, root=0)

        func = crit_func(x_free_econ_eval)
        comm.Send([np.array(func), MPI.DOUBLE], dest=0, tag=rank)

    else:
        raise AssertionError
