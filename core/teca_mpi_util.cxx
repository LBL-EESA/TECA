#include "teca_mpi_util.h"

#include "teca_config.h"

#include <vector>

namespace teca_mpi_util
{
// **************************************************************************
int equipartition_communicator(MPI_Comm comm,
    int new_comm_size, MPI_Comm *new_comm)
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        int rank = 0;
        int n_ranks = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);

        if (n_ranks < new_comm_size)
        {
            // can't increase beyond the original sizew
            return 0;
        }

        int stride = n_ranks / new_comm_size;
        //int in_new_comm = (n_ranks % new_comm_size) == 0;

        // get the ranks in the new commmunicator
        std::vector<int> ranks(new_comm_size);
        for (int i = 0; i < new_comm_size; ++i)
            ranks[i] = i*stride;

        // make a group containing the ranks
        MPI_Group world_group;
        MPI_Comm_group(comm, &world_group);

        MPI_Group new_group;
        MPI_Group_incl(world_group, new_comm_size, ranks.data(), &new_group);

        // create the new communicator
        MPI_Comm_create_group(comm, new_group, 0, new_comm);

        // clean up
        MPI_Group_free(&world_group);
        MPI_Group_free(&new_group);
    }
#endif
    return 0;
}
}
