// file: helloworld.cxx
#include <boost/python.hpp>
#include <mpi4py/mpi4py.h>
#include "helloworld.h"
using namespace boost::python;

static void wrap_sayhello(object py_comm) {
  PyObject* py_obj = py_comm.ptr();
  MPI_Comm *comm_p = PyMPIComm_Get(py_obj);
  if (comm_p == NULL) throw_error_already_set();
  sayhello(*comm_p);
}

BOOST_PYTHON_MODULE(helloworld) {
  if (import_mpi4py() < 0) return;
  def("sayhello", wrap_sayhello);
}
