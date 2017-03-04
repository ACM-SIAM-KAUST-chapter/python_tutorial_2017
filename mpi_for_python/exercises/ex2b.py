from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD

right = (comm.rank + 1) % comm.size
left  = (comm.rank - 1) % comm.size

sendbuf = numpy.arange(comm.rank, comm.rank+1000, dtype="f")
recvbuf_right = numpy.empty(1000, dtype="f")
recvbuf_left  = numpy.empty(1000, dtype="f")

req1 = comm.Isend([sendbuf, MPI.FLOAT], dest=right)
req2 = comm.Isend([sendbuf, MPI.FLOAT], dest=left)
req3 = comm.Irecv([recvbuf_right, MPI.FLOAT], source=right)
req4 = comm.Irecv([recvbuf_left,  MPI.FLOAT], source=left)

MPI.Request.Waitall([req1, req2, req3, req4])

assert recvbuf_right[0] == right
assert recvbuf_left[0]  == left
