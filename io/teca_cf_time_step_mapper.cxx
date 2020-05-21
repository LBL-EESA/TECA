#include "teca_cf_time_step_mapper.h"

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::to_stream(std::ostream &os)
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
    if (rank == 0)
    {
        os << "start_time_step = " << this->start_time_step << std::endl
            << "end_time_step = " << this->end_time_step << std::endl
            << "n_time_steps = " << this->n_time_steps << std::endl
            << "n_time_steps_per_file = " << this->n_time_steps_per_file << std::endl
            << "n_files = " << this->n_files << std::endl
            << "n_ranks = " << n_ranks << std::endl
            << "rank\tfirst_time_step\tlast_time_step" << std::endl;

        for (int i = 0; i < n_ranks; ++i)
            os << i << "\t" << block_start[i] << "\t"
                << block_start[i] + block_size[i] - 1 << std::endl;

        os << "file\tranks" << std::endl;
        for (int i = 0; i < this->n_files; ++i)
        {
            os << i << "\t";
            std::set<int> &f_ranks = this->file_ranks[i];
            std::set<int>::iterator it = f_ranks.begin();
            std::set<int>::iterator end = f_ranks.end();
            for (; it != end; ++it)
            {
                os << *it << ", ";
            }
            os << std::endl;
        }
    }

    return 0;
}

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
        up_reqs.back().set(this->index_request_key, time_step);
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
int teca_cf_time_step_mapper::get_file_id(long time_step, long &file_id)
{
    file_id = time_step/this->n_time_steps_per_file;
    return 0;
}

// --------------------------------------------------------------------------
p_teca_cf_layout_manager teca_cf_time_step_mapper::get_layout_manager(
    long time_step)
{
    long file_id = time_step/this->n_time_steps_per_file;
    file_table_t::iterator it = this->file_table.find(file_id);
    if (it == this->file_table.end())
    {
        TECA_ERROR("No layout manager for time step " << time_step)
        return nullptr;
    }
    return it->second;
}

// --------------------------------------------------------------------------
int teca_cf_time_step_mapper::initialize(MPI_Comm comm, long first_step,
    long last_step, long steps_per_file, const teca_metadata &md)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    this->comm = comm;
    this->start_time_step = first_step;
    this->end_time_step = last_step;
    this->n_time_steps_per_file = steps_per_file;

    // locate the keys that enable us to know how many
    // requests we need to make and what key to use
    if (md.get("index_initializer_key", this->index_initializer_key))
    {
        TECA_ERROR("No time_step initializer key has been specified")
        return -1;
    }

    if (md.get("index_request_key", this->index_request_key))
    {
        TECA_ERROR("No time_step request key has been specified")
        return -1;
    }

    // locate available time_steps
    this->n_time_steps = 0;
    if (md.get(this->index_initializer_key, n_time_steps))
    {
        TECA_ERROR("metadata is missing the initializer key \""
            << this->index_initializer_key << "\"")
        return -1;
    }

    // apply restriction
    long last
        = this->end_time_step >= 0 ? this->end_time_step : n_time_steps - 1;

    long first
        = ((this->start_time_step >= 0) && (this->start_time_step <= last))
            ? this->start_time_step : 0;

    this->n_time_steps = last - first + 1;

    // partition time_steps across MPI ranks. each rank
    // will end up with a unique block of time_steps
    // to process.
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

    // map time_steps to ranks
    long n_big_blocks = this->n_time_steps%n_ranks;
    this->block_size.resize(n_ranks);
    this->block_start.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i)
    {
        this->block_size[i] = 1;
        this->block_start[i] = 0;
        if (i < n_big_blocks)
        {
            this->block_size[i] = this->n_time_steps/n_ranks + 1;
            this->block_start[i] = first + this->block_size[i]*i;
        }
        else
        {
            this->block_size[i] = this->n_time_steps/n_ranks;
            this->block_start[i] = first + this->block_size[i]*i + n_big_blocks;
        }
    }

    // get the number of files to write
    this->n_files = this->n_time_steps / this->n_time_steps_per_file +
        (this->n_time_steps % this->n_time_steps_per_file ? 1 : 0);

    // map file id to ranks
    this->file_ranks.resize(this->n_files);
    int last_file_rank = 0;
    std::vector<int> file_ranks_i;
    file_ranks_i.reserve(n_ranks);
    for (long i = 0; i < this->n_files; ++i)
    {
        long file_time_step_0 = i*this->n_time_steps_per_file;
        long file_time_step_1 = file_time_step_0 + this->n_time_steps_per_file - 1;

        file_ranks_i.clear();

        for (int j = last_file_rank; j < n_ranks; ++j)
        {
            long block_time_step_0 = this->block_start[j];
            long block_time_step_1 = block_time_step_0 + this->block_size[j] - 1;

            // check if this rank is writing to this file
            long bf_int_0 = file_time_step_0 > block_time_step_0 ?
                 file_time_step_0 : block_time_step_0;

            long bf_int_1 = file_time_step_1 < block_time_step_1 ?
                file_time_step_1 : block_time_step_1;
            if (bf_int_0 <= bf_int_1)
            {
                // yes add it to the list
                file_ranks_i.push_back(j);
            }
        }

        last_file_rank = file_ranks_i.size() ? file_ranks_i[0] : 0;

        // store for a later look up
        this->file_ranks[i].insert(file_ranks_i.begin(), file_ranks_i.end());
    }

    // allocate per-file communicators
    this->alloc_file_comms();

    // create layout managers for the local files
    // proceed file by file, this ensures a deterministic non-blocking order
    for (long i = 0; i < this->n_files; ++i)
    {
        MPI_Comm comm_i = this->file_comms[i];
        if (comm_i != MPI_COMM_NULL)
        {
            // this rank will write to this file, create a layout
            // manager that will do the work of putting data on disk
            long first_step = i*this->n_time_steps_per_file;

            long n_steps = first_step + this->n_time_steps_per_file > this->n_time_steps ?
                this->n_time_steps : this->n_time_steps_per_file;

            this->file_table[i] = teca_cf_layout_manager::New(comm_i, i, first_step, n_steps);
        }
    }

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
