from mpi4py import MPI
import numpy

def matvec(comm, A, x):
    m = A.shape[0]
    p = comm.Get_size()
    xg = numpy.zeros(m*p, dtype=x.dtype)
    comm.Allgather(x, xg)
    y = numpy.dot(A, xg)
    return y
    
comm = MPI.COMM_WORLD

n = 10 # local row size

A = numpy.zeros((n, n*comm.size), dtype='d')
diag1 = numpy.arange(n)
diag2 = numpy.arange(n*comm.rank, n*(comm.rank+1))
A[diag1, diag2] = 1

x = numpy.arange(n, dtype=A.dtype)

y = matvec(comm, A, x)
assert numpy.allclose(x, y)
