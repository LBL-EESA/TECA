#include "teca_cf_interval_time_step_mapper.h"

#include <iomanip>

namespace {
// **************************************************************************
int locate(long step, const std::vector<std::pair<long, long>> &brackets,
    size_t l, size_t r, long &i)
{
    size_t m = (l + r) / 2;

    const std::pair<long, long> &br = brackets[m];

    if ((step >= br.first) && (step <= br.second))
    {
        // found
        i = m;
        return 0;
    }
    else if (l == r)
    {
        // not found
        return -1;
    }
    else if (step < br.first)
    {
        // search left
        return locate(step, brackets, l, m, i);
    }
    else if (step > br.second)
    {
        // search right
        if (m == l) m = r;
        return locate(step, brackets, m, r, i);
    }

    // not found
    return -1;
}
}


// --------------------------------------------------------------------------
int teca_cf_interval_time_step_mapper::to_stream(std::ostream &os)
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
            << "n_files = " << this->n_files << std::endl
            << "n_ranks = " << n_ranks << std::endl << std::endl
            << std::left << std::setw(8) << "rank"
            << std::left << std::setw(12) << "first_step"
            << std::left << std::setw(12) << "last_step"
            << std::endl;

        for (int i = 0; i < n_ranks; ++i)
            os << std::left << std::setw(8) << i
                << std::left << std::setw(12) << block_start[i]
                << std::left << std::setw(12) << block_start[i] + block_size[i] - 1
                << std::endl;

        os << std::endl
            << std::left << std::setw(8) << "file"
            << std::left << std::setw(16) << "steps"
            << std::left << std::setw(8) << "n_steps"
            << "ranks" << std::endl;

        long n_steps_total = 0;

        for (int i = 0; i < this->n_files; ++i)
        {
            std::string steps("[");
            steps += std::to_string(this->file_steps[i].first);
            steps += ", ";
            steps += std::to_string(this->file_steps[i].second);
            steps += "]";

            long n_steps = this->file_steps[i].second - this->file_steps[i].first + 1;
            n_steps_total += n_steps;

            os << std::left << std::setw(8) << i
                << std::left << std::setw(16) << steps
                << std::left << std::setw(8) << n_steps;

            std::set<int> &f_ranks = this->file_ranks[i];
            size_t n_f_ranks = f_ranks.size();
            std::set<int>::iterator it = f_ranks.begin();
            std::set<int>::iterator end = f_ranks.end();

            if (n_f_ranks)
            {
                os << "[" << *it;
                ++it;

                for (; it != end; ++it)
                    os << ", " << *it;

                os << "]";
            }

            os << std::endl;
        }

        os << std::endl << "n_steps_total = " << n_steps_total << std::endl;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_interval_time_step_mapper::get_file_id(long time_step, long &file_id)
{
    file_id = -1;
    if (::locate(time_step, this->file_steps, 0, this->n_files - 1, file_id))
    {
        TECA_ERROR("Failed to locate the file id for time step " << time_step)
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
p_teca_cf_layout_manager teca_cf_interval_time_step_mapper::get_layout_manager(
    long time_step)
{
    long file_id = -1;
    if (locate(time_step, this->file_steps, 0, this->n_files - 1, file_id))
    {
        TECA_ERROR("Failed to locate the file id for time step " << time_step)
        return nullptr;
    }

    file_table_t::iterator it = this->file_table.find(file_id);
    if (it == this->file_table.end())
    {
        TECA_ERROR("No layout manager for time step " << time_step)
        return nullptr;
    }

    return it->second;
}

// --------------------------------------------------------------------------
int teca_cf_interval_time_step_mapper::initialize(MPI_Comm comm,
    long first_step, long last_step,
    const teca_calendar_util::p_interval_iterator &it,
    const teca_metadata &md)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    this->comm = comm;
    this->start_time_step = first_step;
    this->end_time_step = last_step;

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

    // enumerate the steps in each file
    if (it->initialize(md, first, last))
    {
        TECA_ERROR("Failed to initialize the interval iterator")
        return -1;
    }

    this->file_steps.clear();
    while (*it)
    {
        teca_calendar_util::time_point first_step;
        teca_calendar_util::time_point last_step;

        it->get_next_interval(first_step, last_step);

        // apply subset on a bracket that intersects it
        first_step.index = first > first_step.index ? first : first_step.index;
        last_step.index = last < last_step.index ? last : last_step.index;

        this->file_steps.emplace_back(
            std::make_pair(first_step.index, last_step.index));
    }

    // make a correction to the restriction because the seasonal iterator only
    // works on full seasons.
    first = this->file_steps[0].first;
    last = this->file_steps.back().second;
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
    long n_big_blocks = this->n_time_steps % n_ranks;
    this->block_size.resize(n_ranks);
    this->block_start.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i)
    {
        this->block_size[i] = 1;
        this->block_start[i] = 0;
        if (i < n_big_blocks)
        {
            this->block_size[i] = this->n_time_steps / n_ranks + 1;
            this->block_start[i] = first + this->block_size[i]*i;
        }
        else
        {
            this->block_size[i] = this->n_time_steps / n_ranks;
            this->block_start[i] = first + this->block_size[i]*i + n_big_blocks;
        }
    }

    // get the number of files to write
    this->n_files = this->file_steps.size();

    // map file id to ranks
    this->file_ranks.resize(this->n_files);
    int last_file_rank = 0;
    std::vector<int> file_ranks_i;
    file_ranks_i.reserve(n_ranks);
    for (long i = 0; i < this->n_files; ++i)
    {
        long file_time_step_0 = this->file_steps[i].first;
        long file_time_step_1 = this->file_steps[i].second;

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
            long first_step = this->file_steps[i].first;
            long last_step = this->file_steps[i].second;

            long n_steps = last_step - first_step + 1;

            this->file_table[i] = teca_cf_layout_manager::New(comm_i, i, first_step, n_steps);
        }
    }

    return 0;
}
