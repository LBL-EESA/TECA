#include "teca_cf_writer.h"

#include "teca_config.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_file_util.h"
#include "teca_netcdf_util.h"
#include "teca_coordinate_util.h"
#include "teca_cf_time_step_mapper.h"
#include "teca_cf_block_time_step_mapper.h"
#include "teca_cf_interval_time_step_mapper.h"
#include "teca_cf_spatial_time_step_mapper.h"
#include "teca_cf_space_time_time_step_mapper.h"
#include "teca_cf_layout_manager.h"
#include "teca_coordinate_util.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif



class teca_cf_writer::internals_t
{
public:
    internals_t() : mapper(), layout_defined(0)
    {}

    p_teca_cf_time_step_mapper mapper;
    int layout_defined;
};



// --------------------------------------------------------------------------
teca_cf_writer::teca_cf_writer() :
    file_name(""), date_format("%F-%HZ"), number_of_spatial_partitions(1),
    partition_x(0), partition_y(1), partition_z(1),
    minimum_block_size_x(1), minimum_block_size_y(1), minimum_block_size_z(1),
    number_of_temporal_partitions(0), temporal_partition_size(0),
    first_step(0), last_step(-1), layout(monthly), partitioner(temporal),
    index_executive_compatability(0), steps_per_file(128),
    mode_flags(NC_CLOBBER|NC_NETCDF4), use_unlimited_dim(0),
    collective_buffer(-1), compression_level(-1), flush_files(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
    this->set_stream_size(1);
    this->internals = new teca_cf_writer::internals_t;
}

// --------------------------------------------------------------------------
teca_cf_writer::~teca_cf_writer()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cf_writer::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cf_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, file_name,
            "A path and file name pattern to write data to. For time varying output"
            " spanning more than one file the string %t% is replaced with the date"
            " and/or time of the first time step in the file. The formatting of the"
            " date/time encoding is specified using --date_format.")
        TECA_POPTS_GET(std::string, prefix, date_format,
            "A strftime format used when encoding dates into the output"
            " file names (%F-%HZ). %t% in the file name is replaced with date/time"
            " of the first time step in the file using this format specifier.")
        TECA_POPTS_GET(long, prefix, first_step,
            "set the first time step to process")
        TECA_POPTS_GET(long, prefix, last_step,
            "Set the last time step to process. A value less than 0 results"
            " in all steps being processed.")
        TECA_POPTS_GET(int, prefix, partitioner,
            "Selects the partitioner to use. May be one of: \"temporal\","
            " \"spatial\", or \"space_time\"")
        TECA_POPTS_GET(int, prefix, index_executive_compatability,
            "If set and spatial partitioner is enabled, the writer will make"
            " one request per time step using the index_request_key as the"
            " teca_index_executive would. This could be used parallelize"
            " existing algorithms over space and time.")
        TECA_POPTS_GET(long, prefix, number_of_spatial_partitions,
            "Sets the number of spatial partitions to create. Use -1 to create one"
            " partition per MPI rank")
        TECA_POPTS_GET(int, prefix, partition_x,
            "Enables/disables spatial partitioning in the x-direction")
        TECA_POPTS_GET(int, prefix, partition_y,
            "Enables/disables spatial partitioning in the y-direction")
        TECA_POPTS_GET(int, prefix, partition_z,
            "Enables/disables spatial partitioning in the z-direction")
        TECA_POPTS_GET(long, prefix, minimum_block_size_x,
            "Sets the minimum block size in the x-direction when partitioning"
            " spatially.")
        TECA_POPTS_GET(long, prefix, minimum_block_size_y,
            "Sets the minimum block size in the y-direction when partitioning"
            " spatially.")
        TECA_POPTS_GET(long, prefix, minimum_block_size_z,
            "Sets the minimum block size in the z-direction when partitioning"
            " spatially.")
        TECA_POPTS_GET(long, prefix, number_of_temporal_partitions,
            "Set the number of temporal partitions. If set to less than one then the"
            " number of time steps is used. The temporal_partition_size property"
            " takes precedence, if it is set then the this property is ignored.")
        TECA_POPTS_GET(long, prefix, temporal_partition_size,
            "Set the size of the temporal partitions. If set to less than one then"
            " the number_of_temporal_partition property is used instead.")
        TECA_POPTS_GET(int, prefix, layout,
            "Set the layout for writing files. May be one of : number_of_steps(1),"
            "  daily(2), monthly(3), seasonal(4), or yearly(5)")
        TECA_POPTS_GET(unsigned int, prefix, steps_per_file,
            "set the number of time steps to write per file")
        TECA_POPTS_GET(int, prefix, mode_flags,
            "mode flags to pass to NetCDF when creating the file")
        TECA_POPTS_GET(int, prefix, use_unlimited_dim,
            "if set the slowest varying dimension is specified to be "
            "NC_UNLIMITED.")
        TECA_POPTS_GET(int, prefix, collective_buffer,
            "When set, enables colective buffering. This can only be used with"
            " the spatial partitoner when the number of MPI ranks is equal to the"
            " number of spatial partitons. A value of -1 can be used to"
            " automatically enbable collective buffering when it is safe to do"
            " so.")
        TECA_POPTS_GET(int, prefix, compression_level,
            "sets the zlib compression level used for each variable;"
            " does nothing if the value is less than or equal to 0.")
        TECA_POPTS_GET(int, prefix, flush_files,
            "if set files are flushed before they are closed.")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, point_arrays,
            "the list of point centered arrays to write")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, information_arrays,
            "the list of non-geometric arrays to write")
        ;

    this->teca_threaded_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, std::string, prefix, date_format)
    TECA_POPTS_SET(opts, long, prefix, first_step)
    TECA_POPTS_SET(opts, long, prefix, last_step)
    TECA_POPTS_SET(opts, int, prefix, partitioner)
    TECA_POPTS_SET(opts, int, prefix, index_executive_compatability)
    TECA_POPTS_SET(opts, long, prefix, number_of_spatial_partitions)
    TECA_POPTS_SET(opts, int, prefix, partition_x)
    TECA_POPTS_SET(opts, int, prefix, partition_y)
    TECA_POPTS_SET(opts, int, prefix, partition_z)
    TECA_POPTS_SET(opts, long, prefix, minimum_block_size_x)
    TECA_POPTS_SET(opts, long, prefix, minimum_block_size_y)
    TECA_POPTS_SET(opts, long, prefix, minimum_block_size_z)
    TECA_POPTS_SET(opts, long, prefix, number_of_temporal_partitions)
    TECA_POPTS_SET(opts, long, prefix, temporal_partition_size)
    TECA_POPTS_SET(opts, int, prefix, layout)
    TECA_POPTS_SET(opts, unsigned int, prefix, steps_per_file)
    TECA_POPTS_SET(opts, int, prefix, mode_flags)
    TECA_POPTS_SET(opts, int, prefix, use_unlimited_dim)
    TECA_POPTS_SET(opts, int, prefix, compression_level)
    TECA_POPTS_SET(opts, int, prefix, flush_files)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, point_arrays)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, information_arrays)
}
#endif

// --------------------------------------------------------------------------
int teca_cf_writer::set_layout(const std::string &mode)
{
    if (mode == "daily")
    {
        this->set_layout(daily);
    }
    else if (mode == "monthly")
    {
        this->set_layout(monthly);
    }
    else if (mode == "seasonal")
    {
        this->set_layout(seasonal);
    }
    else if (mode == "yearly")
    {
        this->set_layout(yearly);
    }
    else if (mode == "number_of_steps")
    {
        this->set_layout(number_of_steps);
    }
    else
    {
        TECA_ERROR("Invalid layout mode \"" << mode << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
const char *teca_cf_writer::get_layout_name() const
{
    const char *ret = "unknown";
    switch (this->layout)
    {
        case teca_cf_writer::number_of_steps:
            ret = "number_of_steps";
            break;
        case teca_cf_writer::daily:
            ret = "daily";
            break;
        case teca_cf_writer::monthly:
            ret = "monthly";
            break;
        case teca_cf_writer::seasonal:
            ret = "seasonal";
            break;
        case teca_cf_writer::yearly:
            ret = "yearly";
            break;
    }
    return ret;
}

// --------------------------------------------------------------------------
const char *teca_cf_writer::get_partitioner_name() const
{
    const char *ret = "unknown";
    switch (this->partitioner)
    {
        case teca_cf_writer::temporal:
            ret = "temporal";
            break;
        case teca_cf_writer::spatial:
            ret = "spatial";
            break;
        case teca_cf_writer::space_time:
            ret = "space_time";
            break;
    }
    return ret;
}

// --------------------------------------------------------------------------
void teca_cf_writer::set_partitioner(const std::string &part)
{
    if (part == "temporal")
    {
        this->set_partitioner_to_temporal();
    }
    else if (part == "spatial")
    {
        this->set_partitioner_to_spatial();
    }
    else if (part == "space_time")
    {
        this->set_partitioner_to_space_time();
    }
    else
    {
        TECA_FATAL_ERROR("Failed to set the partitioner."
            " There is no partitioner named \"" << part << "\"")
    }
}

// --------------------------------------------------------------------------
int teca_cf_writer::flush()
{
    // flush the set of files
    if (this->internals->mapper->file_table_apply([&](MPI_Comm comm,
        long file_id, const p_teca_cf_layout_manager &layout_mgr) -> int
        {
            (void)comm;
            if (layout_mgr->flush())
            {
                TECA_ERROR("Failed to flush file " << file_id)
                return -1;
            }
            return 0;
        }))
    {
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_cf_writer::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cf_writer::get_output_metadata" << std::endl;
#endif
    (void)port;

    int n_ranks = 1;
    MPI_Comm comm = this->get_communicator();
#if defined(TECA_HAS_MPI)
    int rank = 0;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);
    }
#endif

    const teca_metadata &md_in = input_md[0];

    // get incoming pipeline control keys
    std::string up_initializer_key;
    if (md_in.get("index_initializer_key", up_initializer_key))
    {
        TECA_FATAL_ERROR("Invalid metadata. Failed to locate the"
            " index_initializer_key. This indicates a failure in the"
            " upstream execution.")
        return teca_metadata();
    }

    // pass metadata through
    teca_metadata md_out(md_in);

    // hijack executive control keys
    // a number of the I/O operations are MPI collectives, make sure
    // a request is made to each rank.
    md_out.set("index_initializer_key", std::string("number_of_writers"));
    md_out.set("index_request_key", std::string("writer_id"));
    md_out.set("number_of_writers", n_ranks);

    return md_out;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cf_writer::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cf_writer::get_upstream_request" << std::endl;
#endif
    (void) port;

    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
        MPI_Comm_size(this->get_communicator(), &n_ranks);
    }
#endif

    std::vector<teca_metadata> up_reqs;

    // check that the user hadn't forgotten to specify arrays to write.
    if ((this->point_arrays.size() == 0) &&
        (this->information_arrays.size() == 0))
    {
        TECA_FATAL_ERROR("The arrays to write have not been specified")
        return up_reqs;
    }

    // get the executive control keys for the upstream.
    const teca_metadata &md_in = input_md[0];

    // locate the keys that enable us to know how many
    // requests we need to make and what key to use
    std::string index_initializer_key;
    if (md_in.get("index_initializer_key", index_initializer_key))
    {
        TECA_FATAL_ERROR("No index_initializer_key has been specified")
        return up_reqs;
    }

    long n_time_steps = 0;
    if (md_in.get(index_initializer_key, n_time_steps))
    {
        TECA_FATAL_ERROR("Missing index initializer \""
            << index_initializer_key << "\"")
        return up_reqs;
    }

    std::string index_request_key;
    if (md_in.get("index_request_key", index_request_key))
    {
        TECA_FATAL_ERROR("No index_request_key has been specified")
        return up_reqs;
    }

    // apply restriction
    long last_time_step = this->last_step >= 0 ?
         this->last_step : n_time_steps - 1;

    long first_time_step = ((this->first_step >= 0) &&
        (this->first_step <= last_time_step)) ? this->first_step : 0;

    // get the extent describing the size of the output file
    // this can come from a few places: the request takes precedence,
    // either bounds or extent, if either of those are not present
    // then use the incoming metadata. this can have an extent and
    // if not whole exent.
    double bounds[6] = {0.0};
    unsigned long extent[6] = {0ul};
    if (!request.get("bounds", bounds, 6))
    {
        if (teca_coordinate_util::bounds_to_extent(bounds, md_in, extent))
        {
            if (md_in.get("extent", extent, 6))
            {
                if (md_in.get("whole_extent", extent, 6))
                {
                    TECA_FATAL_ERROR("Failed to determine extent to write"
                        " from the requested bounds [" << bounds << "] and "
                        " failed to locate extent and whole_extent metadata")
                    return up_reqs;
                }
            }
        }
    }
    else if (request.get("extent", extent, 6))
    {
        if (md_in.get("extent", extent, 6))
        {
            if (md_in.get("whole_extent", extent, 6))
            {
                TECA_FATAL_ERROR("Failed to determine extent to write from"
                    " the request bounds and extent keys and failed to"
                    " locate extent and whole_extent metadata")
                return up_reqs;
            }
        }
    }

    // initialize the mapper.
    MPI_Comm comm = this->get_communicator();
    if (this->layout == number_of_steps)
    {
        p_teca_cf_block_time_step_mapper bmap =
            teca_cf_block_time_step_mapper::New();

        if (bmap->initialize(comm, first_time_step,
            last_time_step, this->steps_per_file, index_request_key))
        {
            TECA_FATAL_ERROR("Failed to initialize the block mapper")
            return up_reqs;
        }

        this->internals->mapper = bmap;
    }
    else
    {
        // create the requested interval iterator
        teca_calendar_util::p_interval_iterator it =
            teca_calendar_util::interval_iterator_factory::New(this->layout);

        if (!it)
        {
            TECA_FATAL_ERROR("Failed to create an iterator for layout "
                <<  this->layout)
            return up_reqs;
        }

        // initialize the iterator
        if (it->initialize(md_in, first_time_step, last_time_step))
        {
            TECA_FATAL_ERROR("Failed to initialize the interval iterator")
            return up_reqs;
        }

        // initialize the layout mapper
        if (this->partitioner == temporal)
        {
            p_teca_cf_interval_time_step_mapper imap =
                teca_cf_interval_time_step_mapper::New();

            if (imap->initialize(comm, first_time_step, last_time_step,
                it, index_request_key))
            {
                TECA_FATAL_ERROR("Failed to initialize the interval mapper")
                return up_reqs;
            }

            this->internals->mapper = imap;
        }
        else if (this->partitioner == space_time)
        {
            long n_temporal_partitions = (this->index_executive_compatability ?
                0 : this->number_of_temporal_partitions);

            long temporal_partition_size = (this->index_executive_compatability ?
                1 : this->temporal_partition_size);

            p_teca_cf_space_time_time_step_mapper imap =
                teca_cf_space_time_time_step_mapper::New();

            if (imap->initialize(comm, first_time_step, last_time_step,
                n_temporal_partitions, temporal_partition_size, extent,
                this->number_of_spatial_partitions,
                this->partition_x, this->partition_y, this->partition_z,
                this->minimum_block_size_x, this->minimum_block_size_y,
                this->minimum_block_size_z, it,
                this->index_executive_compatability, index_request_key))
            {
                TECA_FATAL_ERROR("Failed to initialize the interval mapper")
                return up_reqs;
            }

            imap->write_partitions();

            this->internals->mapper = imap;
        }
        else if (this->partitioner == spatial)
        {
            long n_temporal_partitions = (this->index_executive_compatability ?
                0 : this->number_of_temporal_partitions);

            long temporal_partition_size = (this->index_executive_compatability ?
                1 : this->temporal_partition_size);

            p_teca_cf_spatial_time_step_mapper imap =
                teca_cf_spatial_time_step_mapper::New();

            if (imap->initialize(comm, first_time_step, last_time_step,
                n_temporal_partitions, temporal_partition_size, extent,
                this->number_of_spatial_partitions,
                this->partition_x, this->partition_y, this->partition_z,
                this->minimum_block_size_x, this->minimum_block_size_y,
                this->minimum_block_size_z, it,
                this->index_executive_compatability, index_request_key))
            {
                TECA_FATAL_ERROR("Failed to initialize the interval mapper")
                return up_reqs;
            }

            imap->write_partitions();

            this->internals->mapper = imap;
        }
        else
        {
            TECA_FATAL_ERROR("Unknown partitioner mode " << this->partitioner)
            return up_reqs;
        }
    }

    // validate the collective buffer setting. this will cause a dead lock if
    // it is enabled at the wrong configuration.
    int use_collective_buffer = this->collective_buffer;

    int collective_buffer_valid = ((this->partitioner == spatial) &&
        ((this->number_of_spatial_partitions < 1) ||
        this->number_of_spatial_partitions == n_ranks));

    if ((this->collective_buffer > 0) && !collective_buffer_valid)
    {
        TECA_FATAL_ERROR("Collective buffering has been enabled for an invalid"
            " configuration (spatial partitioner "
            << (this->partitioner == spatial ? "on" : "off")
            << " " << this->number_of_spatial_partitions << " partitions "
            << n_ranks << " MPI ranks). Collective buffering can be safely used"
            " when the spatial partitioner is enabled and the number of MPI"
            " ranks is equal to the number of spatial partitions")
        return up_reqs;
    }

    // automatically enable collective buffering when using the spatial
    // partitioner and the number of spatial partitons is equal to the
    // number of ranks
    if (this->collective_buffer < 0)
    {
        use_collective_buffer = (collective_buffer_valid ? 1 : 0);
    }

    // create and define the set of files
    if (this->internals->mapper->file_table_apply([&](MPI_Comm comm,
        long file_id, const p_teca_cf_layout_manager &layout_mgr) -> int
        {
            (void)comm;
            // the upstream requests are all queued up. Before issuing them
            // intialize file specific book keeping structure
            if (layout_mgr->create(this->file_name, this->date_format, md_in,
                this->mode_flags, this->use_unlimited_dim))
            {
                TECA_FATAL_ERROR("Failed to create file " << file_id)
                return -1;
            }

            // define the file layout the first time through. This is a collective
            // operation and the global view of the data must be passed.It is assumed
            // that each time step has the same global view.
            if (layout_mgr->define(md_in, extent, this->point_arrays,
                this->information_arrays, use_collective_buffer,
                this->compression_level))
            {
                TECA_FATAL_ERROR("failed to define file " << file_id)
                return -1;
            }

            return 0;
        }))
    {
        return up_reqs;
    }

    if (this->get_verbose())
    {
        if (rank == 0)
        {
            std::ostringstream oss;
            oss << "Configuring the writer for " << this->get_partitioner_name()
                << " partitioning and " << (use_collective_buffer ?
                "collective buffering" : "independent access");

            if (this->partitioner != temporal)
            {
                if (this->index_executive_compatability)
                {
                    oss << " in compatability mode";
                }
            }

            oss << " with a " << this->get_layout_name() << " layout";

            TECA_STATUS(<< oss.str())
        }
        this->internals->mapper->to_stream(std::cerr);
    }


    // construct the base request, pass through incoming request for bounds,
    // arrays, etc...  reset executive control keys
    teca_metadata base_req(request);
    std::set<std::string> arrays;
    base_req.get("arrays", arrays);
    arrays.insert(this->point_arrays.begin(), this->point_arrays.end());
    arrays.insert(this->information_arrays.begin(), this->information_arrays.end());
    base_req.set("arrays", arrays);
    base_req.remove("writer_id");
    base_req.set("index_request_key", index_request_key);

    if (this->internals->mapper->get_upstream_requests(base_req, up_reqs))
    {
        TECA_FATAL_ERROR("Failed to create upstream requests")
        return up_reqs;
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cf_writer::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request, int streaming)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_cf_writer::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int n_ranks = 1;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
        MPI_Comm_size(this->get_communicator(), &n_ranks);
    }
#endif

    // get the number of datasets in hand. these will be written to one of
    // the files, depending on its time step
    long n_indices = input_data.size();
    for (long i = 0; i < n_indices; ++i)
    {
        // set up the write. collect various data and metadata
        const_p_teca_cartesian_mesh in_mesh =
            std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[i]);

        if (!in_mesh)
        {
            if (rank == 0)
                TECA_FATAL_ERROR("input mesh 0 is empty input or not a cartesian mesh")
            return nullptr;
        }

        // get the spatial extent of the incoming data
        unsigned long extent[6] = {0ul};
        if (in_mesh->get_extent(extent))
        {
            TECA_FATAL_ERROR("Failed to determine the spatial extent to be written")
            return nullptr;
        }

        // get the temporal extent of the incoming data
        unsigned long temporal_extent[2] = {0l};
        if ((this->partitioner == temporal) ||
            ((this->partitioner != temporal) && this->index_executive_compatability))
        {
            if (in_mesh->get_time_step(temporal_extent[0]))
            {
                TECA_FATAL_ERROR("Failed to determine the time step to be written")
                return nullptr;
            }
            temporal_extent[1] = temporal_extent[0];
        }
        else if (in_mesh->get_temporal_extent(temporal_extent))
        {
            TECA_FATAL_ERROR("Failed to determine the temporal extent to be written")
            return nullptr;
        }

        // get the layout managers needed to write this extent
        std::vector<p_teca_cf_layout_manager> managers;
        if (this->internals->mapper->get_layout_manager(temporal_extent, managers))
        {
            TECA_FATAL_ERROR("No layout manager found for temporal extent ["
                << temporal_extent << "]")
            return nullptr;
        }

        // give each manager a chance to write the steps it is responsble for
        int n_managers = managers.size();
        for (int j = 0; j < n_managers; ++j)
        {
            const p_teca_cf_layout_manager &layout_mgr = managers[j];

            // write the arrays
            if (layout_mgr->write(extent, temporal_extent,
                in_mesh->get_point_arrays(), in_mesh->get_information_arrays()))
            {
                TECA_FATAL_ERROR("Manager " << j << " of " << n_managers
                    << " failed to write temporal extent [" << temporal_extent
                    << "]")
                return nullptr;
            }

            if (this->verbose > 1)
            {
                std::ostringstream oss;
                oss << "Wrote: extent = [" << extent << "], temporal_extent = ["
                    << temporal_extent << "].";
                //layout_mgr->to_stream(oss);
                TECA_STATUS(<< oss.str())
            }
        }
    }

    // close the file when all data has been written
    if (!streaming)
    {
        if ((this->flush_files && this->flush()) ||
            this->internals->mapper->finalize())
        {
            TECA_FATAL_ERROR("Failed to finalize I/O")
            return nullptr;
        }
    }

    return nullptr;
}
