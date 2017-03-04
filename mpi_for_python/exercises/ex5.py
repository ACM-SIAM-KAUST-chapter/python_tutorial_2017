from mpi4py import MPI
import numpy

x1, x2 = -2.0, 1.0
y1, y2 = -1.0, 1.0
w,  h  = 150*4, 100*4
maxit  = 127

def mandelbrot(x, y, maxit):
    c = x + y*1j
    z = 0 + 0j
    it = 0
    while abs(z) < 2 and it < maxit:
        z = z**2 + c
        it += 1
    return it

def mandelbrot_block(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    # number of rows to compute here
    N = h // size + (h % size > rank)
    # first row to compute here
    start = comm.scan(N)-N
    # local result
    Cl = numpy.empty([N, w], dtype='i')
    # compute owned rows
    time = MPI.Wtime()
    dx = (x2 - x1) / w
    dy = (y2 - y1) / h
    for i in range(N):
        y = y1 + (start+i) * dy
        for j in range(w):
            x = x1 + j * dx
            Cl[i, j] = mandelbrot(x, y, maxit)
    time = MPI.Wtime() - time
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
    return C, time

def mandelbrot_cyclic(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    # number of rows to compute here
    N = h // size + (h % size > rank)
    # indices of rows to compute here
    I = numpy.arange(rank, h, size, dtype='i')
    # local result
    Cl = numpy.empty([N, w], dtype='i')
    # compute owned rows
    time = MPI.Wtime()
    dx = (x2 - x1) / w
    dy = (y2 - y1) / h
    for k in range(N):
        y = y1 + int(I[k]) * dy
        for j in range(w):
            x = x1 + j * dx
            Cl[k, j] = mandelbrot(x, y, maxit)
    time = MPI.Wtime() - time
    # gather results at root
    N = numpy.array(N, dtype='i')
    counts  = None
    result  = None
    if rank == 0:
        counts  = numpy.empty(size, dtype='i')
        result  = numpy.empty([h, w], dtype='i')
    comm.Gather(sendbuf=[N, MPI.INT],
                recvbuf=[counts, MPI.INT],
                root=0)
    rowtype = MPI.INT.Create_contiguous(w).Commit()
    comm.Gatherv(sendbuf=[Cl, MPI.INT],
                 recvbuf=[result, (counts, None), rowtype],
                 root=0)
    rowtype.Free()
    # reconstruct result
    C = None
    if rank == 0:
        C = numpy.empty([h, w], dtype='i')
        indices = []
        for i in range(size):
            idx = numpy.arange(i, h, size, dtype='i')
            indices.append(idx)
        indices = numpy.concatenate(indices)
        C[indices, :] = result
    return C, time


comm = MPI.COMM_WORLD

Cb, Tb = mandelbrot_block(comm)
Cc, Tc = mandelbrot_cyclic(comm)
if comm.rank==0:
    assert numpy.allclose(Cb,Cc)

Tb_max = comm.reduce(Tb, op=MPI.MAX, root=0)
Tb_min = comm.reduce(Tb, op=MPI.MIN, root=0)
if comm.rank==0:
    case = "BLOCK  ->"
    Tmax, Tmin = Tb_max, Tb_min
    print(case, "Tmax=%f"%Tmax,"Tmin=%f"%Tmin,"Tmax/Tmin=%f"%(Tmax/Tmin))

Tc_max = comm.reduce(Tc, op=MPI.MAX, root=0)
Tc_min = comm.reduce(Tc, op=MPI.MIN, root=0)
if comm.rank==0:
    case = "CYCLIC ->"
    Tmax, Tmin = Tc_max, Tc_min
    print(case, "Tmax=%f"%Tmax,"Tmin=%f"%Tmin,"Tmax/Tmin=%f"%(Tmax/Tmin))
