#ifndef teca_mpi_h
#define teca_mpi_h

#include "teca_config.h"

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#else
using MPI_Comm = void*;
#define MPI_COMM_WORLD nullptr
#define MPI_COMM_SELF nullptr
#define MPI_COMM_NULL nullptr
#endif

#endif
