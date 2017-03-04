from mpi4py import MPI
comm = MPI.COMM_WORLD

sendmsg = comm.rank+1

recvmsg = comm.scan(sendmsg, op=MPI.SUM)

import sys
sys.stdout.write("[%d] %s\n" % (comm.rank, recvmsg))
