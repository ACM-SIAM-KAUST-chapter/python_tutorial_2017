program main
  !use mpi
  include 'mpif.h'
  integer :: ierr, rank, size, len
  character (len=MPI_MAX_PROCESSOR_NAME) :: name

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  call MPI_Get_processor_name(name, len, ierr)
  print*, 'Hello, World! ', &
       'I am process ', rank, ' of ', size, &
       ' on ', name(1:len)
  call MPI_Finalize(ierr)
end program main
