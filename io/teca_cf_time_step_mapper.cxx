#include "teca_cf_time_step_mapper.h"

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::get_upstream_requests(
    teca_metadata base_req, std::vector<teca_metadata> &up_reqs)
{
    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &n_ranks);
    }
#endif
    // apply the base request to local time_steps.
    long n_req = this->block_size[rank];
    long first = this->block_start[rank];
    up_reqs.reserve(n_req);
    for (long i = 0; i < n_req; ++i)
    {
        long time_step = i + first;
        up_reqs.push_back(base_req);
        up_reqs.back().set(this->index_request_key, {time_step, time_step});
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::alloc_file_comms()
{
    // create communicator for each file
    this->file_comms.resize(this->n_files, this->comm);
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        int rank = 0;
        int n_ranks = 1;
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &n_ranks);
        if (n_ranks > 1)
        {
            // parallel run. partition ranks to files and make communicators
            // that encode the partitioning.
            for (long i = 0; i < this->n_files; ++i)
            {
                std::set<int> &file_ranks_i = this->file_ranks[i];
                int color = MPI_UNDEFINED;
                if (file_ranks_i.find(rank) != file_ranks_i.end())
                    color = 0;

                MPI_Comm file_comm = MPI_COMM_NULL;
                MPI_Comm_split(this->comm, color, rank, &file_comm);
                this->file_comms[i] = file_comm;
            }
        }
        else
        {
            // serial run. use the only communicator
            for (long i = 0; i < this->n_files; ++i)
            {
                MPI_Comm file_comm = MPI_COMM_NULL;
                MPI_Comm_dup(this->comm, &file_comm);
                this->file_comms[i] = file_comm;
            }
        }
    }
    else
    {
       // mpi is not in use
       for (long i = 0; i < this->n_files; ++i)
           this->file_comms[i] = this->comm;
    }
#endif
    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::free_file_comms()
{
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        for (long i = 0; i < this->n_files; ++i)
        {
            MPI_Comm comm_i = this->file_comms[i];
            if (comm_i != MPI_COMM_NULL)
            {
                MPI_Comm_free(&this->file_comms[i]);
                this->file_comms[i] = MPI_COMM_NULL;
            }
        }
    }
#endif
    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::finalize()
{
    // create layout managers for the local files
    // proceed file by file, this ensures a deterministic non-blocking order
    for (long i = 0; i < this->n_files; ++i)
    {
        MPI_Comm comm_i = this->file_comms[i];
        if (comm_i != MPI_COMM_NULL)
            file_table[i]->close();
    }

    this->free_file_comms();

    this->file_comms.clear();
    this->file_table.clear();

    return 0;
}
