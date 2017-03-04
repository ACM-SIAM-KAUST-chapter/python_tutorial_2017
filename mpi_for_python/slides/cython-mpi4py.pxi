cdef import from "mpi.h":
    ctypedef void* MPI_Comm
    MPI_Comm MPI_COMM_NULL
    MPI_Comm MPI_COMM_SELF
    MPI_Comm MPI_COMM_WORLD
    int MPI_Comm_size(MPI_Comm,int*)
    int MPI_Comm_rank(MPI_Comm,int*)

cdef inline int CHKERR(int ierr) except -1:
    if ierr != 0:
        raise RuntimeError("MPI error code %d" % ierr)
    return 0
