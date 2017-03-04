from mpi4py import MPI

comm = MPI.COMM_WORLD
group = comm.Get_group()

newgroup = group.Incl(range(0, comm.size, 2))
#newgroup = group.Range_incl([(0, comm.size-1, 2)])
newcomm = comm.Create(newgroup)

if comm.rank % 2 == 0:
    assert newcomm.size == comm.size // 2 + comm.size % 2
    assert newcomm.rank == comm.rank // 2
else:
    assert newcomm == MPI.COMM_NULL

group.Free()
newgroup.Free()
if newcomm != MPI.COMM_NULL:
    newcomm.Free()
