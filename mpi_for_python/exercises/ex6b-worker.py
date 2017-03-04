from mpi4py import MPI
import numpy

def mandelbrot(x, y, maxit):
    c = x + y*1j
    z = 0 + 0j
    it = 0
    while abs(z) < 2 and it < maxit:
        z = z**2 + c
        it += 1
    return it

master = MPI.Comm.Get_parent()
if master == MPI.COMM_NULL:
    print("parent communicator is MPI_COMM_NULL")
    MPI.COMM_WORLD.Abort(1)

nprocs = master.Get_size()
myrank = master.Get_rank()

# receive parameters and unpack
rmsg = numpy.zeros(4, dtype='f')
master.Bcast(rmsg, root=0)
x1, x2, y1, y2 = rmsg
imsg = numpy.zeros(3, dtype='i')
master.Bcast(imsg, root=0)
w, h, maxit = imsg

# number of rows to compute here
N = h // nprocs + (h % nprocs > myrank)
N = numpy.array(N, dtype='i')

# indices of rows to compute here
I = numpy.arange(myrank, h, nprocs, dtype='i')

# local result
C = numpy.empty([N, w], dtype='i')

# compute owned rows
dx = (x2 - x1) / w
dy = (y2 - y1) / h
for k in range(N):
    y = y1 + I[k] * dy
    for j in range(w):
        x = x1 + j * dx
        C[k, j] = mandelbrot(x, y, maxit)

# send number of rows computed here
master.Gather (N, None, root=0)

# send number of rows computed here
master.Gatherv(I, None, root=0)

# send data of rows computed here
master.Gatherv(C, None, root=0)

# we are done
master.Disconnect()
