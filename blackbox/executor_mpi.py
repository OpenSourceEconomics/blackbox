import sys
import os

import numpy as np

if 'PMI_SIZE' in os.environ.keys():
    try:
        from mpi4py import MPI
    except ImportError:
        pass


def mpi_executor(batch, num_free):

    return _MpiPool(batch, num_free)


class _MpiPool(object):

    def __init__(self, batch, num_free):
        """This method initializes the pool of MPI slaves."""
        # Start an army of workers for evaluations of the criterion function.
        info = MPI.Info.Create()
        info.update({"wdir": os.getcwd()})

        file_ = os.path.dirname(os.path.realpath(__file__)) + '/executor_mpi_worker.py'
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=[file_], maxprocs=batch, info=info)

        # We need to inform everybody about the number of free parameters.
        num_free = np.array(num_free, dtype='int32')
        comm.Bcast([num_free, MPI.INT], root=MPI.ROOT)

        # TODO: All slaves should send a message that they are ready and then we can also remove
        # the *.pkl.

        # Store class attributes for future references.
        self.attr = dict()
        self.attr['num_free'] = None
        self.attr['batch'] = batch
        self.attr['comm'] = comm

    def terminate(self):
        """This method terminates the pool of workers."""
        cmd = np.array(-1, dtype='int32')
        self.attr['comm'].Bcast([cmd, MPI.INT], root=MPI.ROOT)
        self.attr['comm'].Disconnect()

    def evaluate(self, points):
        """Points i s a nested list"""
        # We want to ensure a speedy execution.
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if self.attr['num_free'] is None:
            self.attr['num_free'] = points.shape[1]

        # Distribute class attributes
        num_free = self.attr['num_free']
        batch = self.attr['batch']
        comm = self.attr['comm']

        # Prepare slaves for evaluation task and send evaluation points.
        cmd = np.array(1, dtype='int32')
        comm.Bcast([cmd, MPI.INT], root=MPI.ROOT)
        comm.Scatter(points, np.empty(num_free), root=MPI.ROOT)

        # Collect information from slaves of value of criterion function.
        funcs = list()
        for rank in range(batch):
            func = np.zeros(1, dtype='float')
            comm.Recv([func, MPI.DOUBLE], source=rank, tag=MPI.ANY_TAG)
            funcs.append(func)

        return funcs
