from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

def get_intervals():
    prompt = "Enter the number of intervals: (0 quits) "
    try:
        n = int(input(prompt))
        if n < 0: n = 0
    except:
        n = 0
    return n

def compute_pi(n, start=0, step=1):
    h = 1.0 / n
    s = 0.0
    for i in range(start, n, step):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h

while True:

    n = None
    if myrank == 0:
        n = get_intervals()
    n = comm.bcast(n, root=0)
    if n == 0:
        break

    mypi = compute_pi(n, myrank, nprocs)

    pi = comm.reduce(mypi, op=MPI.SUM, root=0)

    if myrank == 0:
        error = abs(pi - math.pi)
        print("pi is approximately %.16f, error is %.16f" % (pi, error))
