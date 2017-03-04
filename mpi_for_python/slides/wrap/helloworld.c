/* file: helloworld.c */
#include <stdio.h>
#include "helloworld.h"
void sayhello(MPI_Comm comm)
{
  int size, rank, nlen;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(name, &nlen);
  printf("Hello, World! I am process "
	 "%d of %d on %s.\n", rank, size, name);
}
