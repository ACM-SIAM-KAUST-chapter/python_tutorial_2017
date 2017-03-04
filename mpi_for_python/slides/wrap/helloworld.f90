! file: helloworld.f90
subroutine sayhello(comm)
  use mpi
  integer :: comm, rank, size, nlen, ierr
  character(len=MPI_MAX_PROCESSOR_NAME) :: name
  call MPI_Comm_size(comm, size, ierr)
  call MPI_Comm_rank(comm, rank, ierr)
  call MPI_Get_processor_name(name, nlen, ierr)
  print *, 'Hello, World! I am process ', &
            rank,' of ',size, ' on ', name, '.'
end subroutine sayhello
