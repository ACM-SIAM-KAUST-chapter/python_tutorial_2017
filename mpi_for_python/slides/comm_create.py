from mpi4py import MPI

comm = MPI.COMM_WORLD
group = comm.Get_group()

newgroup = group.Excl([0])
newcomm = comm.Create(newgroup)

if comm.rank == 0:
    assert newcomm == MPI.COMM_NULL
else:
    assert newcomm.size == comm.size - 1
    assert newcomm.rank == comm.rank - 1

group.Free(); newgroup.Free()
if newcomm: newcomm.Free()
