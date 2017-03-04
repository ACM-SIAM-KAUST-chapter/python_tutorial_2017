from mpi4py import MPI
comm = MPI.COMM_WORLD

sendmsg = comm.rank

recvmsg = comm.allreduce(sendmsg, op=MPI.SUM)

import sys
sys.stdout.write("[%d] %s\n" % (comm.rank, recvmsg))
