include "cython-mpi4py.pxi"

cdef class Comm:
    cdef MPI_Comm ob_mpi
    ...
    def Get_size(self):
        cdef int size
        CHKERR( MPI_Comm_size(self.ob_mpi, &size) )
        return size
    def Get_rank(self):
        cdef int rank
        CHKERR( MPI_Comm_rank(self.ob_mpi, &rank) )
        return rank
    ...

cdef inline Comm NewComm(MPI_Comm comm_c):
    cdef Comm comm_py = Comm()
    comm_py.ob_mpi = comm_c
    return comm_py

COMM_NULL  = NewComm(MPI_COMM_NULL)
COMM_SELF  = NewComm(MPI_COMM_SELF)
COMM_WORLD = NewComm(MPI_COMM_WORLD)
