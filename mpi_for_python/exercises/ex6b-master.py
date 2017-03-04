# $ mpiexec -n 1 python ex6b-master.py 5

from mpi4py import MPI
import sys, numpy

x1, x2 = -2.0, 1.0
y1, y2 = -1.0, 1.0
w,  h  = 150, 100
maxit  = 127

# spawn worker
nworkers = int(sys.argv[1])
worker = MPI.COMM_SELF.Spawn(sys.executable,
                             ['ex6b-worker.py'],
                             maxprocs=nworkers)
size = worker.Get_remote_size()

# send parameters
rmsg = numpy.array([x1, x2, y1, y2], dtype='f')
imsg = numpy.array([w, h, maxit], dtype='i')
worker.Bcast(rmsg, root=MPI.ROOT)
worker.Bcast(imsg, root=MPI.ROOT)

# gather results
counts  = numpy.empty(size, dtype='i')
indices = numpy.empty(h, dtype='i')
result  = numpy.empty([h, w], dtype='i')
worker.Gather(sendbuf=None,
              recvbuf=[counts, MPI.INT],
              root=MPI.ROOT)
worker.Gatherv(sendbuf=None,
               recvbuf=[indices, (counts, None), MPI.INT],
               root=MPI.ROOT)
rowtype = MPI.INT.Create_contiguous(w).Commit()
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
