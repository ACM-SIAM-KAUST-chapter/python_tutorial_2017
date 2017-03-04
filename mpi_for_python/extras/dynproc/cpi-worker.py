#!/usr/bin/env python
from mpi4py import MPI
from array import array

parent = MPI.Comm.Get_parent()
rank = parent.Get_rank()
size = parent.Get_size()

while True:

    n = array('i', [0])
    if rank == 0:
        parent.Recv(n, 0)

    MPI.COMM_WORLD.Bcast(n, 0)
    n = n[0]
    if n < 1: break

    h = 1.0/n
    s = 0.0
    for i in range(rank+1, n+1, size):
        x = h * (i - 0.5)
        s += 4 / (1 + x*x)

    pi_local = array('d', [s * h])
    pi = array('d', [0])
    MPI.COMM_WORLD.Reduce(pi_local, pi, root=0)

    if rank == 0:
        parent.Send(pi, 0)

parent.Disconnect()
