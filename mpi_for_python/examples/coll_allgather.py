from mpi4py import MPI
comm = MPI.COMM_WORLD

sendmsg = comm.rank**2

recvmsg = comm.allgather(sendmsg)

import sys
sys.stdout.write("[%d] %s\n" % (comm.rank, recvmsg))
