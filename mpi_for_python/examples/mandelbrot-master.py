# $ mpif90 -o mandelbrot-worker.exe mandelbrot-worker.f90
# $ mpiexec -n 1 python mandelbrot-master.py 4


from mpi4py import MPI
import numpy
import sys

x1, x2 = -2.0, 1.0
y1, y2 = -1.0, 1.0
w,  h  = 1200, 800
maxit  = 127

import os
dirname = os.path.abspath(os.path.dirname(__file__))
executable = os.path.join(dirname, 'mandelbrot-worker.exe')

# spawn worker
nworkers = int(sys.argv[1])
worker = MPI.COMM_SELF.Spawn(executable, maxprocs=nworkers)
size = worker.Get_remote_size()

# send parameters
rmsg = numpy.array([x1, x2, y1, y2], dtype='f')
imsg = numpy.array([w, h, maxit], dtype='i')
worker.Bcast([rmsg, MPI.REAL], root=MPI.ROOT)
worker.Bcast([imsg, MPI.INTEGER], root=MPI.ROOT)

# gather results
counts  = numpy.empty(size, dtype='i')
indices = numpy.empty(h, dtype='i')
result  = numpy.empty([h, w], dtype='i')
worker.Gather(sendbuf=None,
              recvbuf=[counts, MPI.INTEGER],
              root=MPI.ROOT)
worker.Gatherv(sendbuf=None,
               recvbuf=[indices, (counts, None), MPI.INTEGER],
               root=MPI.ROOT)
rowtype = MPI.INTEGER.Create_contiguous(w).Commit()
worker.Gatherv(sendbuf=None,
               recvbuf=[result, (counts, None), rowtype],
               root=MPI.ROOT)
rowtype.Free()

# disconnect worker
worker.Disconnect()

# reconstruct result
C = numpy.empty([h, w], dtype='i')
C[indices, :] = result

try:
    from matplotlib import pyplot as plt
    plt.imshow(C, aspect='equal')
    plt.spectral()
    if not plt.isinteractive():
        import signal
        def action(*args): raise SystemExit
        signal.signal(signal.SIGALRM, action)
        signal.alarm(2)
    plt.show()
except:
    pass
