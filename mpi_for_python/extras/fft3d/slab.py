from mpi4py import MPI
import numpy as np
import pyfftw


def _subsize(N, size, rank):
    return N // size + (N % size > rank)


def _distribution(N, size):
    q = N // size
    r = N % size
    n = s = i = 0
    while i < size:
        n = q
        s = q * i
        if i < r:
            n += 1
            s += i
        else:
            s += r
        yield n, s
        i += 1


class Slab(object):

    def __init__(self, comm, shape, dtype=float, **options):
        self.comm = MPI.COMM_NULL
        self._subarraysA = []
        self._subarraysB = []
        self._fft2 = self._ifft2 = None
        self._fft1 = self._ifft1 = None

        assert isinstance(comm, MPI.Intracomm)
        self.comm = comm
        size = comm.Get_size()
        rank = comm.Get_rank()

        M, N, P = shape
        if np.issubdtype(dtype, np.floating):
            assert P % 2 == 0
            Q = P//2 + 1
        else:
            Q = P
        assert M >= size
        assert N >= size

        opts = dict(
            avoid_copy=True,
            overwrite_input=True,
            auto_align_input=True,
            auto_contiguous=True,
            threads=1,
        )
        opts.update(options)
        dtype = np.dtype(dtype)
        empty = pyfftw.empty_aligned
        fft1 = pyfftw.builders.fft
        ifft1 = pyfftw.builders.ifft
        if np.issubdtype(dtype, np.floating):
            fft2 = pyfftw.builders.rfftn
            ifft2 = pyfftw.builders.irfftn
            opts_ifft2 = opts.copy()
            del opts_ifft2['overwrite_input']
        else:
            fft2 = pyfftw.builders.fftn
            ifft2 = pyfftw.builders.ifftn
            opts_ifft2 = opts

        m = _subsize(M, size, rank)
        dtype = np.dtype(dtype)
        U = empty([m,N,P], dtype=dtype)
        self._fft2 = fft2(U, axes=(1,2), **opts)
        ctype = self._fft2.output_array.dtype
        F = self._fft2.output_array
        self._ifft2 = ifft2(F, axes=(1,2), **opts_ifft2)
        self._ifft2.update_arrays(F, U)

        n = _subsize(N, size, rank)
        U = empty([M,n,Q], dtype=ctype)
        self._fft1 = fft1(U, axis=0, **opts)
        F = self._fft1.output_array
        self._ifft1 = ifft1(F, axis=0, **opts)
        self._ifft1.update_arrays(F, U)

        datatype = MPI._typedict[ctype.char]
        self._subarraysA = [
            datatype.Create_subarray([M,n,Q], [l,n,Q], [s,0,0]).Commit()
            for l, s in _distribution(M, size)
        ]
        self._subarraysB = [
            datatype.Create_subarray([m,N,Q], [m,l,Q], [0,s,0]).Commit()
            for l, s in _distribution(N, size)
        ]
        self._counts_displs = ([1] * size, [0] * size)

    def destroy(self):
        for t in self._subarraysA:
            t.Free()
        for t in self._subarraysB:
            t.Free()
        self.comm = MPI.COMM_NULL
        self._subarraysA = []
        self._subarraysB = []

    def forward(self, U=None, U_hat=None):
        if U is not None:
            self.forward_input_array[...] = U

        self._fft2()
        self.comm.Alltoallw(
            [self._fft2.output_array, self._counts_displs, self._subarraysB],
            [self._fft1.input_array,  self._counts_displs, self._subarraysA])
        self._fft1()

        out = self.forward_output_array
        if U_hat is not None:
            U_hat[...] = out
        else:
            U_hat = out
        return U_hat

    def backward(self, U_hat=None, U=None, normalise_idft=True):
        if U_hat is not None:
            self.backward_input_array[...] = U_hat

        self._ifft1(normalise_idft=normalise_idft)
        self.comm.Alltoallw(
            [self._ifft1.output_array, self._counts_displs, self._subarraysA],
            [self._ifft2.input_array,  self._counts_displs, self._subarraysB])
        self._ifft2(normalise_idft=normalise_idft)

        out = self.backward_output_array
        if U is not None:
            U[...] = out
        else:
            U = out
        return U

    @property
    def forward_input_array(self):
        return self._fft2.input_array

    @property
    def forward_output_array(self):
        return self._fft1.output_array

    @property
    def backward_input_array(self):
        return self._ifft1.input_array

    @property
    def backward_output_array(self):
        return self._ifft2.output_array
