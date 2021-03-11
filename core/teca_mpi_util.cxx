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
            // can't increase beyond the original size
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

// **************************************************************************
int split_communicator(MPI_Comm world_comm,
    int group_size, MPI_Comm *group_comm)
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        int world_rank = 0;
        int world_size = 1;

        MPI_Comm_rank(world_comm, &world_rank);
        MPI_Comm_size(world_comm, &world_size);

        MPI_Group world_group = MPI_GROUP_EMPTY;
        MPI_Comm_group(world_comm, &world_group);

        int group_id = world_rank / group_size;
        int group_start = group_id * group_size;
        int group_end = std::min(world_size, group_start + group_size);
        int group_range[3] = {group_start, group_end, 1};

        MPI_Group sub_group = MPI_GROUP_EMPTY;
        MPI_Group_range_incl(world_group, 1, &group_range, &sub_group);

        MPI_Comm_create(world_comm, sub_group, group_comm);

        MPI_Group_free(&world_group);
        MPI_Group_free(&sub_group);
    }
#endif
    return 0;
}

// **************************************************************************
int mpi_rank_0(MPI_Comm comm)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
    }
#endif
    if (rank == 0)
        return 1;
    return 0;
}
}
