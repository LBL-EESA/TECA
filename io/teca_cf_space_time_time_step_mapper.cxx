#include "teca_cf_space_time_time_step_mapper.h"
#include "teca_coordinate_util.h"

//#define VISUALIZE_PARTITIONS
#if defined(VISUALIZE_PARTITIONS)
#include "teca_vtk_util.h"
#endif

#include <iomanip>

using namespace teca_coordinate_util;


// --------------------------------------------------------------------------
int teca_cf_space_time_time_step_mapper::to_stream(std::ostream &os)
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
        // dump the domain decomposition
        os << "start_time_step = " << this->start_time_step << std::endl
            << "end_time_step = " << this->end_time_step << std::endl
            << "n_time_steps = " << this->n_time_steps << std::endl
            << "n_files = " << this->n_files << std::endl
            << "n_ranks = " << n_ranks << std::endl << std::endl
            << std::left << std::setw(8) << "rank"
            << std::left << std::setw(16) << "temporal part."
            << "spatial part." << std::endl;

        unsigned long n_spatial_partitions = this->spatial_partitions.size();
        unsigned long n_temporal_partitions = this->temporal_partitions.size();
        unsigned long n_partitions = n_spatial_partitions * n_temporal_partitions;

        long n_big = n_partitions % n_ranks;

        for (int i = 0; i < n_ranks; ++i)
        {
            // compute rank i's work assignment
            unsigned long blk_size = n_partitions / n_ranks + (i < n_big ? 1 : 0);
            unsigned long blk_start = blk_size*i + (i < n_big ? 0 : n_big);

            // generate a request for each piece of work
            for (unsigned long q = 0; q < blk_size; ++q)
            {
                unsigned long qq = blk_start + q;

                unsigned long t = qq / n_spatial_partitions;
                unsigned long s = qq % n_spatial_partitions;

                std::ostringstream ostp;
                ostp << "[" << this->temporal_partitions[t] << "]";

                std::ostringstream ossp;
                ossp << "[" << this->spatial_partitions[s] << "]";

                os << std::left << std::setw(8) << i
                    << std::left << std::setw(16) << ostp.str()
                    << ossp.str() << std::endl;
            }
        }

        // dump the file table
        os << std::endl
            << std::left << std::setw(8) << "file"
            << std::left << std::setw(16) << "steps"
            << std::left << std::setw(8) << "n_steps"
            << "ranks" << std::endl;

        unsigned long n_steps_total = 0;

        for (int i = 0; i < this->n_files; ++i)
        {
            std::string steps("[");
            steps += std::to_string(this->file_steps[i].first);
            steps += ", ";
            steps += std::to_string(this->file_steps[i].second);
            steps += "]";

            unsigned long n_steps = this->file_steps[i].second - this->file_steps[i].first + 1;
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
int teca_cf_space_time_time_step_mapper::get_file_id(long time_step, long &file_id)
{
    file_id = -1;
    if (find_extent_containing_step(time_step, this->file_steps,  file_id))
    {
        TECA_ERROR("Failed to locate the file id for time step " << time_step)
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
p_teca_cf_layout_manager
teca_cf_space_time_time_step_mapper::get_layout_manager(long time_step)
{
    long file_id = -1;
    if (find_extent_containing_step(time_step, this->file_steps,  file_id))
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
int teca_cf_space_time_time_step_mapper::get_layout_manager(
    const unsigned long temporal_extent[2],
    std::vector<p_teca_cf_layout_manager> &managers)
{
    // find the manager responsible for the first step in the range
    unsigned long q = temporal_extent[0];
    while (q <= temporal_extent[1])
    {
        // find the file id for this time step
        long file_id = -1;
        if (find_extent_containing_step(q, this->file_steps,  file_id))
        {
            TECA_ERROR("Failed to locate the file id for time step " << q)
            return -1;
        }

        // using the file id get the manager
        file_table_t::iterator it = this->file_table.find(file_id);
        if (it == this->file_table.end())
        {
            TECA_ERROR("No layout manager for time step " << q)
            return -1;
        }

        // add to the list of managers to return
        managers.push_back(it->second);

        // the next time step to look for is the first one not stored in this file
        q = this->file_steps[file_id].second + 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_space_time_time_step_mapper::initialize(MPI_Comm comm,
    unsigned long first_step, unsigned long last_step,
    unsigned long number_of_temporal_partitions,
    unsigned long temporal_partition_size, unsigned long *wextent,
    unsigned long number_of_spatial_partitions,
    int partition_x, int partition_y, int partition_z,
    unsigned long min_block_size_x, unsigned long min_block_size_y,
    unsigned long min_block_size_z,
    const teca_calendar_util::p_interval_iterator &it,
    int index_executive_compatability, const std::string &index_request_key)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
#endif
    this->comm = comm;
    this->start_time_step = first_step;
    this->end_time_step = last_step;
    this->index_executive_compatability = index_executive_compatability;
    this->index_request_key = index_request_key;
    this->whole_extent = as_spatial_extent(wextent);

    // partition work across MPI ranks. each rank will end up with a
    // unique subset of the data to process.
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

    // enumerate the steps in each file
    this->file_steps.clear();
    while (*it)
    {
        teca_calendar_util::time_point interval[2];

        it->get_next_interval(interval[0], interval[1]);

        // apply subset on a bracket that intersects it
        interval[0].index = std::max<long>(first_step,  interval[0].index);
        interval[1].index = std::min<long>(last_step, interval[1].index);

        this->file_steps.emplace_back(
            std::make_pair(interval[0].index, interval[1].index));
    }

    // make a correction to the restriction because the seasonal iterator only
    // works on full seasons.
    first_step = this->file_steps[0].first;
    last_step = this->file_steps.back().second;
    this->n_time_steps = last_step - first_step + 1;

    // partiton the time dimension
    if (partition({first_step, last_step},
        number_of_temporal_partitions, temporal_partition_size,
        this->temporal_partitions))
    {
        TECA_ERROR("Failed to partition the temporal domain")
        return -1;
    }

    // partition the spatial domain. if not set then partition such that each
    // rank has a unique subset of the domain
    if (number_of_spatial_partitions < 1)
    {
        // set things up such that each rank has one chunk of work
        number_of_spatial_partitions = n_ranks / this->n_time_steps +
            (n_ranks % n_time_steps ? 1 : 0);
    }

    if (partition(this->whole_extent, number_of_spatial_partitions,
        partition_x, partition_y, partition_z, min_block_size_x,
        min_block_size_y, min_block_size_z, this->spatial_partitions))
    {
        TECA_ERROR("Failed to partition the spatial domain")
        return -1;
    }

    // map time steps to ranks
    unsigned long n_spatial_partitions = this->spatial_partitions.size();
    unsigned long n_temporal_partitions = this->temporal_partitions.size();
    unsigned long n_partitions = n_spatial_partitions * n_temporal_partitions;

    long n_reg = n_partitions / n_ranks;
    long n_big = n_partitions % n_ranks;

    if ((rank == 0) && (n_reg == 1) && n_big)
    {
        TECA_WARNING("Potential load imbalance with " << n_ranks
            << " ranks and " << n_partitions << " partitions " << n_ranks - n_big
            << " ranks are idle during the final pipeline pass")
    }

    this->block_size.resize(n_ranks);
    this->block_start.resize(n_ranks);
    for (int i = 0; i < n_ranks; ++i)
    {
        // compute rank rank's work assignment
        unsigned long blk_size = n_reg + (i < n_big ? 1 : 0);
        unsigned long blk_start = blk_size*i + (i < n_big ? 0 : n_big);

        // get the time step range associated with this ranks blocks
        unsigned long part_0 = blk_start / n_spatial_partitions;
        unsigned long part_1 = (blk_start + blk_size - 1) / n_spatial_partitions;

        unsigned long step_0 = this->temporal_partitions[part_0][0];
        unsigned long step_1 = this->temporal_partitions[part_1][1];

        // record the rank's step range
        this->block_size[i] = step_1 - step_0 + 1;
        this->block_start[i] = step_0;
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
        unsigned long file_time_step_0 = this->file_steps[i].first;
        unsigned long file_time_step_1 = this->file_steps[i].second;

        file_ranks_i.clear();

        for (int j = last_file_rank; j < n_ranks; ++j)
        {
            unsigned long block_time_step_0 = this->block_start[j];
            unsigned long block_time_step_1 = block_time_step_0 + this->block_size[j] - 1;

            // check if this rank is writing to this file
            unsigned long bf_int_0 = file_time_step_0 > block_time_step_0 ?
                 file_time_step_0 : block_time_step_0;

            unsigned long bf_int_1 = file_time_step_1 < block_time_step_1 ?
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
    //this->alloc_file_comms(n_spatial_partitions);
    this->alloc_file_comms();

    // create layout managers for the local files.  proceed file by file, this
    // ensures a deterministic non-blocking order
    for (long i = 0; i < this->n_files; ++i)
    {
        MPI_Comm comm_i = this->file_comms[i];
        if (comm_i != MPI_COMM_NULL)
        {
            // this rank will write to this file, create a layout
            // manager that will do the work of putting data on disk
            unsigned long first_step = this->file_steps[i].first;
            unsigned long last_step = this->file_steps[i].second;

            unsigned long n_steps = last_step - first_step + 1;

            this->file_table[i] =
                teca_cf_layout_manager::New(comm_i, i, first_step, n_steps);
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_space_time_time_step_mapper::get_upstream_requests(teca_metadata base_req,
        std::vector<teca_metadata> &up_reqs)
{
    // partition work across MPI ranks. each rank will end up with a
    // unique subset of the data to process.
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

    // map time steps to ranks
    unsigned long n_spatial_partitions = this->spatial_partitions.size();
    unsigned long n_temporal_partitions = this->temporal_partitions.size();
    unsigned long n_partitions = n_spatial_partitions * n_temporal_partitions;

    long n_big = n_partitions % n_ranks;

    // compute rank rank's work assignment
    unsigned long blk_size = n_partitions / n_ranks + (rank < n_big ? 1 : 0);
    unsigned long blk_start = blk_size*rank + (rank < n_big ? 0 : n_big);

    // generate a request for each piece of work
    for (unsigned long q = 0; q < blk_size; ++q)
    {
        teca_metadata req(base_req);

        unsigned long qq = blk_start + q;

        unsigned long t = qq / n_spatial_partitions;
        unsigned long s = qq % n_spatial_partitions;

        // request a spatial subset
        req.set("extent", this->spatial_partitions[s]);

        // request a temporal subset
        const temporal_extent_t &temporal_extent = this->temporal_partitions[t];
        if (this->index_executive_compatability)
        {
            if (temporal_extent[0] != temporal_extent[1])
            {
                TECA_ERROR("Can't request multiple time steps [" << temporal_extent
                    << "] in index_executive_compatabilty mode")
                return -1;
            }
        }

        req.set(this->index_request_key, temporal_extent);
        up_reqs.emplace_back(std::move(req));
    }

    return 0;
}

// --------------------------------------------------------------------------
void teca_cf_space_time_time_step_mapper::write_partitions()
{
#if defined(VISUALIZE_PARTITIONS)
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

    // rank 0 will write all partitions
    if (rank != 0) return;

    size_t nx = this->whole_extent[1] - this->whole_extent[0] + 1;
    size_t ny = this->whole_extent[3] - this->whole_extent[2] + 1;

    teca_vtk_util::partition_writer pw(
        "space_time_time_step_mapper_partitions.vtk",
        0.0, -90.0, 0.0, 360.0 / (nx - 1), 180.0 / (ny - 1), 1.0);

    // map time steps to ranks
    unsigned long n_spatial_partitions = this->spatial_partitions.size();
    unsigned long n_temporal_partitions = this->temporal_partitions.size();
    unsigned long n_partitions = n_spatial_partitions * n_temporal_partitions;

    long n_big = n_partitions % n_ranks;

    for (int  i = 0; i < n_ranks; ++i)
    {
        // compute i i's work assignment
        unsigned long blk_size = n_partitions / n_ranks + (i < n_big ? 1 : 0);
        unsigned long blk_start = blk_size*i + (i < n_big ? 0 : n_big);
        for (unsigned long q = 0; q < blk_size; ++q)
        {
            unsigned long qq = blk_start + q;

            unsigned long t = qq / n_spatial_partitions;
            unsigned long s = qq % n_spatial_partitions;

            pw.add_partition(this->spatial_partitions[s].data(),
                this->temporal_partitions[t].data(), i);
        }
    }

    pw.write();
#endif
}
