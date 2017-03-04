from mpi4py import MPI

world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()

color = world_rank % 2
if color == 0:
    key = world_rank
else:
    key = world_size-world_rank

newcomm = MPI.COMM_WORLD.Split(color, key)

rank = newcomm.Get_rank()
size = newcomm.Get_size()

print("[%d] color:%d key:%d -- newcomm: rank=%d size=%d" %
      (world_rank, color, key, rank, size))

newcomm.Free()
