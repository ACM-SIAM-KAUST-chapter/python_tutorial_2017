from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

msg = ("Hello, World! "
       "I am process %d of %d on %s\n" % 
       (rank, size, name))

import sys
sys.stdout.write(msg)
