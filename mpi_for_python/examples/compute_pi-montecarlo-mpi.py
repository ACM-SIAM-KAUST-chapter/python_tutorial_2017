from mpi4py import MPI
import numpy

def compute_pi(samples):
    count = 0
    for x, y in samples:
        if x**2 + y**2 <= 1:
            count += 1
    pi = 4*float(count)/len(samples)
    return pi

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

if myrank == 0:
    N = 100000 // nprocs
    samples = numpy.random.random((nprocs, N, 2))
else:
    samples = None
samples = comm.scatter(samples, root=0)

mypi = compute_pi(samples) / nprocs

pi = comm.reduce(mypi, root=0)

if myrank == 0:
    error = abs(pi - numpy.pi)
    print("pi is approximately %.16f, error is %.16f" % (pi, error))
