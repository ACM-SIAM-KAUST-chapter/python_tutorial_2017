from mpi4py import MPI

comm = MPI.COMM_WORLD
assert comm.size == 2

if comm.rank == 0:
    sendmsg = 777
    src = dst = 1
else:
    sendmsg = "abc"
    src = dst = 0
    
req = comm.isend(sendmsg, dest=dst)
recvmsg = comm.recv(source=src)
req.Wait()

import sys
sys.stdout.write("[%d] sendmsg:%s recvmsg:%s\n" % 
                 (comm.rank, sendmsg, recvmsg))
