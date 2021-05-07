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
    file_name(""), date_format("%F-%HZ"), first_step(0), last_step(-1),
    layout(monthly), steps_per_file(128), mode_flags(NC_CLOBBER|NC_NETCDF4),
    use_unlimited_dim(0), compression_level(-1), flush_files(0)
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

    this->teca_algorithm::get_properties_description(prefix, opts);

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
        TECA_ERROR("Failed to locate index_initializer_key")
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

    std::vector<teca_metadata> up_reqs;

    // get the executive control keys for the upstream.
    const teca_metadata &md_in = input_md[0];

    std::string up_initializer_key;
    if (md_in.get("index_initializer_key", up_initializer_key))
    {
        TECA_ERROR("Failed to locate index_initializer_key")
        return up_reqs;
    }

    std::string up_request_key;
    if (md_in.get("index_request_key", up_request_key))
    {
        TECA_ERROR("Failed to locate index_request_key")
        return up_reqs;
    }

    long n_indices_up = 0;
    if (md_in.get(up_initializer_key, n_indices_up))
    {
        TECA_ERROR("Missing index initializer \"" << up_initializer_key << "\"")
        return up_reqs;
    }

    // check that the user hadn't forgotten to specify arrays to write.
    if ((this->point_arrays.size() == 0) &&
        (this->information_arrays.size() == 0))
    {
        TECA_ERROR("The arrays to write have not been specified")
        return up_reqs;
    }

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
                    TECA_ERROR("Failed to determine extent to write")
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
                TECA_ERROR("Failed to determine extent to write")
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

        if (bmap->initialize(comm, this->first_step,
            this->last_step, this->steps_per_file, md_in))
        {
            TECA_ERROR("Failed to initialize the block mapper")
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
            TECA_ERROR("Failed to create an iterator for layout "
                <<  this->layout)
            return up_reqs;
        }

        // initialize the layout mapper
        p_teca_cf_interval_time_step_mapper imap =
            teca_cf_interval_time_step_mapper::New();

        if (imap->initialize(comm,
            this->first_step, this->last_step, it, md_in))
        {
            TECA_ERROR("Failed to initialize the interval mapper")
            return up_reqs;
        }

        this->internals->mapper = imap;
    }

    if (this->get_verbose())
        this->internals->mapper->to_stream(std::cerr);

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
                TECA_ERROR("Failed to create file " << file_id)
                return -1;
            }

            // define the file layout the first time through. This is a collective
            // operation and the global view of the data must be passed.It is assumed
            // that each time step has the same global view.
            if (layout_mgr->define(md_in, extent, this->point_arrays,
                this->information_arrays, this->compression_level))
            {
                TECA_ERROR("failed to define file " << file_id)
                return -1;
            }

            return 0;
        }))
    {
        return up_reqs;
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
    base_req.set("index_request_key", up_request_key);
    if (this->internals->mapper->get_upstream_requests(base_req, up_reqs))
    {
        TECA_ERROR("Failed to create upstream requests")
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
                TECA_ERROR("input mesh 0 is empty input or not a cartesian mesh")
            return nullptr;
        }

        unsigned long time_step = 0;
        in_mesh->get_time_step(time_step);

        // get the layout manager for this time step, repsonsible for putting the
        // data on disk.
        p_teca_cf_layout_manager layout_mgr =
            this->internals->mapper->get_layout_manager(time_step);

        if (!layout_mgr)
        {
            TECA_ERROR("No layout manager found for time step " << time_step)
            return nullptr;
        }

        // write the arrays
        if (layout_mgr->write(time_step, in_mesh->get_point_arrays(),
            in_mesh->get_information_arrays()))
        {
            TECA_ERROR("Write time step " << time_step << " failed for time step")
            return nullptr;
        }

        if (this->verbose > 1)
        {
            std::ostringstream oss;
            layout_mgr->to_stream(oss);
            TECA_STATUS(<< oss.str())
        }
    }

    // close the file when all data has been written
    if (!streaming)
    {
        if ((this->flush_files && this->flush()) ||
            this->internals->mapper->finalize())
        {
            TECA_ERROR("Failed to finalize I/O")
            return nullptr;
        }
    }

    return nullptr;
}
