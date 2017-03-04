from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Bcast
if rank == 0:
    data = numpy.arange(4, dtype='f')
else:
    data = numpy.zeros(4, dtype='f')
comm.Bcast([data, MPI.FLOAT], root=0)
assert numpy.allclose(data, numpy.arange(4, dtype='f'))

# Scatter
if rank == 0:
    sdata = numpy.arange(size*6, dtype='f')
else:
    sdata = None
rdata = numpy.zeros(6, dtype='f')
comm.Scatter([sdata, MPI.FLOAT], [rdata, MPI.FLOAT], root=0)
assert numpy.allclose(rdata, numpy.arange(rank*6, (rank+1)*6, dtype='f'))

# Gather
sdata = numpy.arange(rank*6, (rank+1)*6, dtype='f')
if rank == 0:
    rdata = numpy.zeros(size*6, dtype='f')
else:
    rdata = None
comm.Gather([sdata, MPI.FLOAT], [rdata, MPI.FLOAT], root=0)
if rank ==0:
    assert numpy.allclose(rdata, numpy.arange(size*6, dtype='f'))
    
