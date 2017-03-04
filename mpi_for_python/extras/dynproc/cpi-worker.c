#include <mpi.h>

int compute_pi(MPI_Comm comm, int n, double *pi)
{
  int i, rank, size;
  double h, s, pi_local;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Bcast(&n, 1, MPI_INT, 0, comm);
  if (n < 1) return 1;

  h = 1.0/n; s = 0.0;
  for (i = rank+1; i < n+1; i += size) {
    double x = h * (i - 0.5);
    s += 4.0 / (1.0 + x*x);
  }
  pi_local = s * h;

  MPI_Reduce(&pi_local, pi, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  return 0;
}

int main(int argc, char *argv[])
{
  MPI_Comm parent;
  int      rank;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_get_parent(&parent);
  MPI_Comm_rank(parent, &rank);

  while (1) {
    int n, stop; double pi;

    if (rank == 0)
      MPI_Recv(&n, 1, MPI_INT, 0, 0, parent, MPI_STATUS_IGNORE);

    stop = compute_pi(MPI_COMM_WORLD, n, &pi);
    if (stop) break;

    if (rank == 0)
      MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, parent);
  }

  MPI_Comm_disconnect(&parent);
  MPI_Finalize();
  return 0;
}
