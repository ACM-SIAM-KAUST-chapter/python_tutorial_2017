from mpi4py import MPI
from array import array
import sys, math

command = sys.argv[1]
child = MPI.COMM_SELF.Spawn(command, [], 8)

def send(n):
    n = array('i', [n])
    child.Send(n, 0)

def recv():
    pi = array('d', [0])
    child.Recv(pi, 0)
    return pi[0]
