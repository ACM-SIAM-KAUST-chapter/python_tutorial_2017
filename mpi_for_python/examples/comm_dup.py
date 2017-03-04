from mpi4py import MPI

comm = MPI.COMM_WORLD
newcomm = comm.Dup()

msg = ("comm: rank=%d size=%d -- "
       "newcomm: rank=%d size=%d\n" %
       (comm.rank, comm.size, 
        newcomm.rank, newcomm.size))

import sys
sys.stdout.write(msg) # print(msg)

newcomm.Free()
