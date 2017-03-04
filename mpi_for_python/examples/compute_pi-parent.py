from mpi4py import MPI
import sys, numpy

worker = MPI.COMM_SELF.Spawn(
    sys.executable,
    args=['compute_pi-child.py'],
    maxprocs=5)

N = numpy.array(10, 'i')
worker.Bcast([N, MPI.INT], root=MPI.ROOT)

PI = numpy.array(0.0, 'd')
worker.Reduce(None, [PI, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)

worker.Disconnect()

error = abs(PI - numpy.pi)
print ("pi is approximately %.16f, "
       "error is %.16f" % (PI, error))
