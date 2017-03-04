from mpi4py import MPI

comm = MPI.COMM_WORLD
assert comm.size == 2

if comm.rank == 0:
    sendmsg = 777
    comm.send(sendmsg, dest=1)
    recvmsg = comm.recv(source=1)
else:
    recvmsg = comm.recv(source=0)
    sendmsg = "abc"
    comm.send(sendmsg, dest=0)

import sys
sys.stdout.write("[%d] sendmsg:%s recvmsg:%s\n" % 
                 (comm.rank, sendmsg, recvmsg))
