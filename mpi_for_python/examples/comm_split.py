from mpi4py import MPI

world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()

if world_rank < world_size//2:
    color = 55
    key = world_size-world_rank
else:
    color = 77
    key = world_rank

newcomm = MPI.COMM_WORLD.Split(color, key)

rank = newcomm.Get_rank()
size = newcomm.Get_size()

msg = ("[%d] color:%d key:%d -- newcomm: rank=%d size=%d\n" % 
       (world_rank, color, key, rank, size))

import sys
sys.stdout.write(msg) # print(msg)

newcomm.Free()
