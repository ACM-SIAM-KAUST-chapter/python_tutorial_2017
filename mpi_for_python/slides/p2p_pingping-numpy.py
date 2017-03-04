from mpi4py import MPI
import numpy
comm = MPI.COMM_WORLD
assert comm.size == 2

if comm.rank == 0:
    array1 = numpy.arange(10000, dtype='f')
    array2 = numpy.empty(10000, dtype='f')
    target = 1
else:
    array1 = numpy.ones(10000, dtype='f')
    array2 = numpy.empty(10000, dtype='f')
    target = 0

request = comm.Isend([array1, MPI.FLOAT], dest=target)
comm.Recv([array2, MPI.FLOAT], source=target)
request.Wait()
