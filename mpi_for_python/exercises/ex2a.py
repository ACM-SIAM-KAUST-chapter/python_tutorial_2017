from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
assert comm.size == 2


if comm.rank == 0:
    sendbuf = numpy.arange(0, 1000, dtype=numpy.float64)
    comm.Send([sendbuf, MPI.DOUBLE], dest=1)
    recvbuf = numpy.empty(1000, dtype=numpy.int32)
    comm.Recv([recvbuf, MPI.INT], source=1)
else:
    recvbuf = numpy.empty(1000, dtype=numpy.float64)
    comm.Recv([recvbuf, MPI.DOUBLE], source=0)
    sendbuf = numpy.arange(0, 1000, dtype=numpy.int32)
    comm.Send([sendbuf, MPI.INT], dest=0)

assert numpy.allclose(sendbuf, recvbuf)
