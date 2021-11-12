#ifndef teca_mpi_util_h
#define teca_mpi_util_h

/// @file

#include "teca_config.h"
#include "teca_mpi.h"

/// Codes dealing with MPI
namespace teca_mpi_util
{
/** Subset the the communicator comm into a new communicator with new_comm_size
 * ranks. ranks are selected from comm with a uniform stride spreading them
 * approximatelyt equally across nodes.
 */
TECA_EXPORT
int equipartition_communicator(MPI_Comm comm,
    int new_comm_size, MPI_Comm *new_comm);

/** Split the communicator into a number of new communicators such that each
 * new communicator has group_size ranks.
 */
TECA_EXPORT
int split_communicator(MPI_Comm comm,
    int group_size, MPI_Comm *new_comm);

/// return non-zero if this process is MPI rank 0
TECA_EXPORT
int mpi_rank_0(MPI_Comm comm);
};

#endif
