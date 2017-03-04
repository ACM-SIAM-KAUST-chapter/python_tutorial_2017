from mpi4py import MPI
import numpy as np
from pencil import Pencil

comm = MPI.COMM_WORLD
M, N, P = 32, 32, 32
fft = Pencil(comm, (M, N, P))

shape = fft.forward_input_array.shape
U = np.random.random(shape)
F = fft.forward(U)
V = fft.backward(F)

assert np.allclose(U, V)
