PROGRAM main

  !USE mpi
  implicit none
  include 'mpif.h'

  integer parent, rank, size, i, n, ierr
  double precision h, s, x, pi_local, pi

  call MPI_INIT(ierr)
  call MPI_COMM_GET_PARENT(parent, ierr)
  call MPI_COMM_RANK(parent, rank, ierr)
  call MPI_COMM_SIZE(parent, size, ierr)

  DO WHILE ( .TRUE. )
     if (rank .EQ. 0) then
        call MPI_RECV(n, 1, MPI_INTEGER, 0, 0, parent, MPI_STATUS_IGNORE, ierr)
     end if

     call MPI_BCAST(n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
     if (n .LT. 1) EXIT

     h = 1 / DFLOAT(n);  s = 0
     DO i = rank+1, n, size
        x = h * (DFLOAT(i) - 0.5)
        s = s +  4 / (1 + x*x)
     END DO
     pi_local = s * h

     call MPI_REDUCE(pi_local, pi, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

     if (rank .EQ. 0) then
        call MPI_SEND(pi, 1, MPI_DOUBLE_PRECISION, 0, 0, parent, ierr)
     end if
  END DO

  call MPI_COMM_DISCONNECT(parent, ierr)
  call MPI_FINALIZE(ierr)

END PROGRAM main
