from mpi4py import MPI
import numpy as np
from pencil import Pencil

def test_pencil():
    from itertools import product
    comm = MPI.COMM_WORLD
    sizes = (7, 8, 9, 16, 17, 24)
    types = 'fdgFDG'
    atol = dict(f=1e-6, d=1e-14, g=1e-15)
    for typecode in types:
        for M, N, P in product(sizes, sizes, sizes):
            if P % 2: continue
            fft = Pencil(comm, (M,N,P), typecode)
            shape = fft.forward_input_array.shape
            dtype = fft.forward_input_array.dtype
            U = np.random.random(shape).astype(dtype)
            if 0:
                F = fft.forward(U)
                V = fft.backward(F)
            else:
                fft.forward_input_array[...] = U
                fft.forward()
                fft.backward()
                V = fft.backward_output_array

            fft.destroy()
            assert np.allclose(U, V, atol=atol[typecode.lower()], rtol=0)


if __name__ == '__main__':
    test_pencil()
