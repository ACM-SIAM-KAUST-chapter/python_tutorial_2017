from mpi4py import MPI
comm = MPI.COMM_WORLD
assert comm.size == 2

if comm.rank == 0:
    sendmsg = 777
    target = 1
else:
    sendmsg = "abc"
    target = 0
    
request = comm.isend(sendmsg, dest=target, tag=77)
recvmsg = comm.recv(source=target, tag=77)
request.wait()
