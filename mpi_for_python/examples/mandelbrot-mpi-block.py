from mpi4py import MPI
import numpy

x1, x2 = -2.0, 1.0
y1, y2 = -1.0, 1.0
w,  h  = 150, 100
maxit  = 127

def mandelbrot(x, y, maxit):
    c = x + y*1j
    z = 0 + 0j
    it = 0
    while abs(z) < 2 and it < maxit:
        z = z**2 + c
        it += 1
    return it

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# number of rows to compute here
N = h // size + (h % size > rank)
# first row to compute here
start = comm.scan(N)-N
# local result
Cl = numpy.empty([N, w], dtype='i')
# compute owned rows
dx = (x2 - x1) / w
dy = (y2 - y1) / h
for i in range(N):
    y = y1 + (start+i) * dy
    for j in range(w):
        x = x1 + j * dx
        Cl[i, j] = mandelbrot(x, y, maxit)

# gather results at root
C = None
counts = comm.gather(N, root=0)
if rank == 0:
    C = numpy.empty([h, w], dtype='i')
rowtype = MPI.INT.Create_contiguous(w).Commit()
comm.Gatherv(sendbuf=[Cl, MPI.INT],
             recvbuf=[C, (counts, None), rowtype],
             root=0)
rowtype.Free()

if rank == 0:
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

MPI.COMM_WORLD.Barrier()
