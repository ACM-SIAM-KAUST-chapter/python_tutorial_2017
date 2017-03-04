from mpi4py import MPI
assert MPI.COMM_WORLD.Get_size() == 2 

win = MPI.Win.Allocate_shared(12, comm=MPI.COMM_WORLD)

mem0, _ = win.Shared_query(0)
mem1, _ = win.Shared_query(1)

import numpy as np
a0 = np.frombuffer(mem0, dtype='i')
a1 = np.frombuffer(mem1, dtype='i')
