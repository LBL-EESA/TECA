%include "teca_mpi.h"
#if defined(TECA_HAS_MPI)
%include <mpi4py/mpi4py.i>
%mpi4py_typemap(Comm, MPI_Comm);
#endif
