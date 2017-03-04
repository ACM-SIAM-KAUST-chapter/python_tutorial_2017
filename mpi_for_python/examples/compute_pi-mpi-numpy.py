from mpi4py import MPI
import numpy

def compute_pi(n, start=0, step=1):
    h = 1.0 / n
    s = 0.0
    for i in range(start, n, step):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

if myrank == 0:
    n = 10
    n = numpy.array(n, dtype='i')
else:
    n = numpy.array(0, dtype='i')
comm.Bcast([n, MPI.INT], root=0)

mypi = compute_pi(n, myrank, nprocs)

mypi = numpy.array(mypi, dtype='d')
if myrank == 0:
    pi = numpy.array(0, dtype='d')
else:
    pi = None
comm.Reduce([mypi, MPI.DOUBLE], 
            [pi, MPI.DOUBLE],
            op=MPI.SUM, root=0)

if myrank == 0:
    error = abs(pi - numpy.pi)
    print("pi is approximately %.16f, error is %.16f" % (pi, error))
