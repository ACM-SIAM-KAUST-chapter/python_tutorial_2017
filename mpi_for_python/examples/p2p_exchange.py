from mpi4py import MPI

comm = MPI.COMM_WORLD

right = (comm.rank + 1) % comm.size
left  = (comm.rank - 1) % comm.size

sendmsg = [comm.rank] * 3

req1 = comm.isend(sendmsg, dest=right)
req2 = comm.isend(sendmsg, dest=left)
lmsg = comm.recv(source=left)
rmsg = comm.recv(source=right)

MPI.Request.Waitall([req1, req2])
assert lmsg == [left] * 3
assert rmsg == [right] * 3

import sys
sys.stdout.write("[%d] left=%s right=%s\n" %
                 (comm.rank, lmsg, rmsg))
