from mpi4py import MPI

world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()

if world_rank < world_size//2:
    color = 55
    key = -world_rank
else:
    color = 77
    key = +world_rank

newcomm = MPI.COMM_WORLD.Split(color, key)
# ...
newcomm.Free()
