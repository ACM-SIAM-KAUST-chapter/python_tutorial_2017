from mpi4py import MPI
comm = MPI.COMM_WORLD

sendmsg = comm.rank

recvmsg = comm.reduce(sendmsg, op=MPI.SUM, root=0)

import sys
sys.stdout.write("[%d] %s\n" % (comm.rank, recvmsg))
