#ifndef teca_mpi_util_h
#define teca_mpi_util_h

#include "teca_mpi.h"

namespace teca_mpi_util
{
// subset the the communicator comm into a new communicator with
// new_comm_size ranks. ranks are selected from comm with a uniform
// stride spreading them approximatelyt equally across nodes.
int equipartition_communicator(MPI_Comm comm,
    int new_comm_size, MPI_Comm *new_comm);

};

#endif
