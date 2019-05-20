#include "teca_system_interface.h"
#include "teca_common.h"
#include "teca_mpi_manager.h"

#include <iostream>
using std::cerr;
using std::endl;


int main(int argc, char **argv)
{
    teca_mpi_manager mpiman(argc, argv);
    teca_system_interface::set_stack_trace_on_mpi_error();

    int rank = mpiman.get_comm_rank();
    int n_ranks = mpiman.get_comm_size();

    // rank 0 tries to send to a non-existent rank
    if (rank == 0)
    {
        int i = 0;
        MPI_Send(&i, 1, MPI_INT, n_ranks, 1234, MPI_COMM_WORLD);
    }

    return 0;
}
