from mpi4py import MPI

comm = MPI.COMM_WORLD
newcomm = comm.Dup()

assert newcomm.size == comm.size
assert newcomm.rank == comm.rank
assert newcomm != comm

result = MPI.Comm.Compare(comm, newcomm)
assert result != MPI.IDENT
assert result == MPI.CONGRUENT
