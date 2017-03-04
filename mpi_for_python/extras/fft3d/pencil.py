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


class Pencil(object):

    def __init__(self, comm, shape, dtype=float, **options):
        self.comm = MPI.COMM_NULL
        self._comm1 = MPI.COMM_NULL
        self._comm2 = MPI.COMM_NULL
        self._subarrays1A = []
        self._subarrays1B = []
        self._subarrays2A = []
        self._subarrays2B = []
        self._fft1 = self._ifft1 = None
        self._fft2 = self._ifft2 = None
        self._fft3 = self._ifft3 = None

        assert isinstance(comm, MPI.Intracomm)
        self.comm = comm
        size = comm.Get_size()
        rank = comm.Get_rank()

        size1, size2 = MPI.Compute_dims(size, 2)
        self._comm1 = comm.Split(rank % size2)
        self._comm2 = comm.Split(rank // size2)
        rank1 = self._comm1.Get_rank()
        rank2 = self._comm2.Get_rank()

        M, N, P = shape
        if np.issubdtype(dtype, np.floating):
            assert P % 2 == 0
            Q = P//2 + 1
        else:
            Q = P
        assert M >= size1
        assert N >= size1
        assert N >= size2
        assert Q >= size2

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
        fft2 = pyfftw.builders.fft
        ifft2 = pyfftw.builders.ifft
        if np.issubdtype(dtype, np.floating):
            fft3 = pyfftw.builders.rfft
            ifft3 = pyfftw.builders.irfft
        else:
            fft3 = pyfftw.builders.fft
            ifft3 = pyfftw.builders.ifft

        m = _subsize(M, size1, rank1)
        n = _subsize(N, size2, rank2)
        dtype = np.dtype(dtype)
        U = empty([m,n,P], dtype=dtype)
        self._fft3 = fft3(U, axis=2, **opts)
        ctype = self._fft3.output_array.dtype
        F = self._fft3.output_array
        self._ifft3 = ifft3(F, axis=2, **opts)
        self._ifft3.update_arrays(F, U)

        m = _subsize(M, size1, rank1)
        q = _subsize(Q, size2, rank2)
        U = empty([m,N,q], dtype=ctype)
        self._fft2 = fft2(U, axis=1, **opts)
        F = self._fft2.output_array
        self._ifft2 = ifft2(F, axis=1, **opts)
        self._ifft2.update_arrays(F, U)

        n = _subsize(N, size1, rank1)
        q = _subsize(Q, size2, rank2)
        U = empty([M,n,q], dtype=ctype)
        self._fft1 = fft1(U, axis=0, **opts)
        F = self._fft1.output_array
        self._ifft1 = ifft1(F, axis=0, **opts)
        self._ifft1.update_arrays(F, U)

        datatype = MPI._typedict[ctype.char]

        m = _subsize(M, size1, rank1)
        n = _subsize(N, size1, rank1)
        q = _subsize(Q, size2, rank2)
        self._subarrays1A = [
            datatype.Create_subarray([M,n,q], [l,n,q], [s,0,0]).Commit()
            for l, s in _distribution(M, size1)
        ]
        self._subarrays1B = [
            datatype.Create_subarray([m,N,q], [m,l,q], [0,s,0]).Commit()
            for l, s in _distribution(N, size1)
        ]
        self._counts_displs1 = ([1] * size1, [0] * size1)

        m = _subsize(M, size1, rank1)
        n = _subsize(N, size2, rank2)
        q = _subsize(Q, size2, rank2)
        self._subarrays2A = [
            datatype.Create_subarray([m,N,q], [m,l,q], [0,s,0]).Commit()
            for l, s in _distribution(N, size2)
        ]
        self._subarrays2B = [
            datatype.Create_subarray([m,n,Q], [m,n,l], [0,0,s]).Commit()
            for l, s in _distribution(Q, size2)
        ]
        self._counts_displs2 = ([1] * size2, [0] * size2)

    def destroy(self):
        if self._comm1:
            self._comm1.Free()
        if self._comm2:
            self._comm2.Free()
        for t in self._subarrays1A:
            t.Free()
        for t in self._subarrays1B:
            t.Free()
        for t in self._subarrays2A:
            t.Free()
        for t in self._subarrays2B:
            t.Free()
        self.comm = MPI.COMM_NULL
        self._comm1 = MPI.COMM_NULL
        self._comm2 = MPI.COMM_NULL
        self._subarrays1A = []
        self._subarrays1B = []
        self._subarrays2A = []
        self._subarrays2B = []

    def forward(self, U=None, U_hat=None):
        if U is not None:
            self.forward_input_array[...] = U

        self._fft3()
        self._comm2.Alltoallw(
            [self._fft3.output_array, self._counts_displs2, self._subarrays2B],
            [self._fft2.input_array,  self._counts_displs2, self._subarrays2A])
        self._fft2()
        self._comm1.Alltoallw(
            [self._fft2.output_array, self._counts_displs1, self._subarrays1B],
            [self._fft1.input_array,  self._counts_displs1, self._subarrays1A])
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
        self._comm1.Alltoallw(
            [self._ifft1.output_array, self._counts_displs1, self._subarrays1A],
            [self._ifft2.input_array,  self._counts_displs1, self._subarrays1B])
        self._ifft2(normalise_idft=normalise_idft)
        self._comm2.Alltoallw(
            [self._ifft2.output_array, self._counts_displs2, self._subarrays2A],
            [self._ifft3.input_array,  self._counts_displs2, self._subarrays2B])
        self._ifft3(normalise_idft=normalise_idft)

        out = self.backward_output_array
        if U is not None:
            U[...] = out
        else:
            U = out
        return U

    @property
    def forward_input_array(self):
        return self._fft3.input_array

    @property
    def forward_output_array(self):
        return self._fft1.output_array

    @property
    def backward_input_array(self):
        return self._ifft1.input_array

    @property
    def backward_output_array(self):
        return self._ifft3.output_array
