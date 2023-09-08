#include "teca_cf_reader.h"
#include "teca_array_attributes.h"
#include "teca_cartesian_mesh.h"
#include "teca_cf_time_axis_data.h"
#include "teca_cf_time_axis_reader.h"
#include "teca_cf_time_axis_data_reduce.h"
#include "teca_dataset_capture.h"
#include "teca_mpi_util.h"
#include "teca_file_util.h"
#include "teca_coordinate_util.h"
#include "teca_netcdf_util.h"
#include "teca_system_util.h"
#include "teca_calcalcs.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata_util.h"

#include <netcdf.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <utility>
#include <memory>
#include <iomanip>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_OPENSSL)
#include <openssl/sha.h>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

using std::endl;
using std::cerr;

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

// internals for the cf reader
class teca_cf_reader_internals
{
public:
    teca_cf_reader_internals()
    {}

    teca_metadata metadata;
};


// --------------------------------------------------------------------------
teca_cf_reader::teca_cf_reader() :
    files_regex(""),
    x_axis_variable("lon"),
    y_axis_variable("lat"),
    z_axis_variable(""),
    t_axis_variable("time"),
    calendar(""),
    t_units(""),
    filename_time_template(""),
    periodic_in_x(0),
    periodic_in_y(0),
    periodic_in_z(0),
    max_metadata_ranks(1024),
    clamp_dimensions_of_one(0),
    collective_buffer(0),
    internals(new teca_cf_reader_internals)
{}

// --------------------------------------------------------------------------
teca_cf_reader::~teca_cf_reader()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cf_reader::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cf_reader":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, file_names,
            "An explcict list of files to read. If specified takes precedence"
            " over --files_regex. Use one of --files_regex or --file_names")
        TECA_POPTS_GET(std::string, prefix, files_regex,
            "A POSIX basic regular expression that matches the set of files to process."
            " Only the final component in a path may conatin a regular expression."
            " Use one of --files_regex or --file_names ")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "name of variable that has x axis coordinates")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "name of variable that has y axis coordinates")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "name of variable that has z axis coordinates. If left empty the"
            " output mesh will be 2D.")
        TECA_POPTS_GET(std::string, prefix, t_axis_variable,
            "name of variable that has time axis coordinates (time). Set to an empty"
            " string to enable override methods (--filename_time_template, --t_values)"
            " or to disable time coordinates completely")
        TECA_POPTS_GET(std::string, prefix, calendar,
            "An optional calendar override. May be one of: standard, Julian,"
            " proplectic_Julian, Gregorian, proplectic_Gregorian, Gregorian_Y0,"
            " proplectic_Gregorian_Y0, noleap, no_leap, 365_day, 360_day. When the"
            " override is provided it takes precedence over the value found in the"
            " file. Otherwise the calendar is expected to be encoded in the data"
            " files using CF2 conventions.")
        TECA_POPTS_GET(std::string, prefix, t_units,
            "An optional CF2 time units specification override declaring the"
            " units of the time axis and a reference date and time from which the"
            " time values are relative to. If this is provided it takes precedence"
            " over the value found in the file. Otherwise the time units are"
            " expected to be encouded in the files using the CF2 conventions")
        TECA_POPTS_GET(std::string, prefix, filename_time_template,
            "An optional std::get_time template string for decoding time from the input"
            " file names. If no calendar is specified the standard calendar is used. If"
            " no units are specified then \"days since %Y-%m-%d 00:00:00\" where Y,m,d"
            " are determined from the filename of the first file. Set t_axis_variable to"
            " an empty string to use.")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, t_values,
            "An optional explicit list of double precision values to use as the"
            " time axis. If provided these take precedence over the values found"
            " in the files. Otherwise the variable pointed to by the t_axis_variable"
            " provides the time values. Set t_axis_variable to an empty string"
            " to use.")
        TECA_POPTS_GET(int, prefix, periodic_in_x,
            "the dataset has a periodic boundary in the x direction")
        TECA_POPTS_GET(int, prefix, periodic_in_y,
            "the dataset has a periodic boundary in the y direction")
        TECA_POPTS_GET(int, prefix, periodic_in_z,
            "the dataset has a periodic boundary in the z direction")
        TECA_POPTS_GET(int, prefix, max_metadata_ranks,
            "set the max number of MPI ranks for reading metadata")
        TECA_POPTS_GET(int, prefix, clamp_dimensions_of_one,
            "If set clamp requested axis extent in where the request is out of"
            " bounds and the coordinate array dimension is 1.")
        TECA_POPTS_GET(int, prefix, collective_buffer,
            "When set, enables colective buffering. This can only be used with"
            " the spatial partitoner when the number of MPI ranks is equal to the"
            " number of spatial partitons and running with a single thread.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, file_names)
    TECA_POPTS_SET(opts, std::string, prefix, files_regex)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, calendar)
    TECA_POPTS_SET(opts, std::string, prefix, t_units)
    TECA_POPTS_SET(opts, std::string, prefix, filename_time_template)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, t_values)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_x)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_y)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_z)
    TECA_POPTS_SET(opts, int, prefix, max_metadata_ranks)
    TECA_POPTS_SET(opts, int, prefix, clamp_dimensions_of_one)
    TECA_POPTS_SET(opts, int, prefix, collective_buffer)
}
#endif

// --------------------------------------------------------------------------
void teca_cf_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cf_reader::clear_cached_metadata()
{
    this->internals->metadata.clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_cf_reader::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_reader::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    // return cached metadata. cache is cleared if
    // any of the algorithms properties are modified
    if (this->internals->metadata)
        return this->internals->metadata;


    // TODO -- look for cached metadata here

    int rank = 0;
    int n_ranks = 1;

#if defined(TECA_HAS_MPI)
    MPI_Comm comm = this->get_communicator();

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);
    }
#endif

    // build a pipeline that will read the time axis in parallel.
    p_teca_cf_time_axis_reader read_time_axis = teca_cf_time_axis_reader::New();
    read_time_axis->set_files_regex(this->files_regex);
    read_time_axis->set_file_names(this->file_names);
    read_time_axis->set_t_axis_variable(this->t_axis_variable);

    // one can limit the number of MPI ranks doing I/O in case of
    // large scale run. once this pipeline runs rank 0 will have
    // the full time axis.
    const_p_teca_cf_time_axis_data time_axis_data;
    if (!t_axis_variable.empty())
    {
        // limit the number of ranks here. this will partition the communicator
        // such that ranks in the new communicator are selected uniformly from
        // the original larger communicator.
        MPI_Comm time_reader_comm = comm;
#if defined(TECA_HAS_MPI)
        if (is_init && (n_ranks > this->max_metadata_ranks))
        {
            if (rank == 0)
            {
                TECA_STATUS("Using " << this->max_metadata_ranks << " of "
                    << n_ranks << " MPI ranks  to read the time axis")
            }

            teca_mpi_util::equipartition_communicator(comm,
                this->max_metadata_ranks, &time_reader_comm);
        }
#endif
        read_time_axis->set_communicator(time_reader_comm);

        p_teca_cf_time_axis_data_reduce reduce_time_axis = teca_cf_time_axis_data_reduce::New();
        reduce_time_axis->set_communicator(time_reader_comm);
        reduce_time_axis->set_input_connection(read_time_axis->get_output_port());
        reduce_time_axis->set_verbose(0);
        // threading does not help because HDF5 calls must be serialized
        reduce_time_axis->set_thread_pool_size(1);

        p_teca_dataset_capture capture_time_axis = teca_dataset_capture::New();
        capture_time_axis->set_communicator(time_reader_comm);
        capture_time_axis->set_input_connection(reduce_time_axis->get_output_port());

        // run the pipeline, rank 0 will have the complete time axis
        capture_time_axis->update();

        time_axis_data = std::dynamic_pointer_cast<const teca_cf_time_axis_data>
            (capture_time_axis->get_dataset());

        if ((rank == 0) && !time_axis_data)
        {
            TECA_FATAL_ERROR("Failed to read the time axis")
            return teca_metadata();
        }
    }
    else
    {
        // skip reading time axis but we still need to process the files.
        // we can do this by running the report phase of the pipeline. only
        // rank 0 needs the list of files.
        if (rank == 0)
            read_time_axis->set_communicator(MPI_COMM_SELF);
        else
            read_time_axis->set_communicator(MPI_COMM_NULL);

        // this will populate the list of files.
        read_time_axis->update_metadata();
    }

    // only rank 0 will parse the dataset. once
    // parsed metadata is broadcast to all
    teca_binary_stream stream;

    int root_rank = 0;
    if (rank == root_rank)
    {
        const std::string &path = read_time_axis->get_path();
        const std::vector<std::string> &files = read_time_axis->get_files();
        size_t n_files = files.size();

        if (n_files == 0)
        {
            TECA_FATAL_ERROR("No files found")
            return teca_metadata();
        }

        int ierr = 0;
        std::string file = path + PATH_SEP + files[0];

        // open the file
        teca_netcdf_util::netcdf_handle fh;
        if (fh.open(file.c_str(), NC_NOWRITE))
        {
            TECA_FATAL_ERROR("Failed to open " << file << endl << nc_strerror(ierr))
            return teca_metadata();
        }

        // enumerate mesh arrays and their attributes
        int n_vars = 0;
        teca_metadata atrs;
        std::vector<std::string> vars;
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if (((ierr = nc_inq_nvars(fh.get(), &n_vars)) != NC_NOERR))
        {
            this->clear_cached_metadata();
            TECA_FATAL_ERROR(
                << "Failed to get the number of variables in file \""
                << file << "\"" << endl
                << nc_strerror(ierr))
            return teca_metadata();
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        for (int i = 0; i < n_vars; ++i)
        {
            std::string name;
            teca_metadata atts;

            if (teca_netcdf_util::read_variable_attributes(fh, i,
                this->x_axis_variable, this->y_axis_variable,
                this->z_axis_variable, this->t_axis_variable,
                this->clamp_dimensions_of_one, name, atts))
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to read " << i <<"th variable attributes")
                return teca_metadata();
            }

            vars.push_back(name);
            atrs.set(name, atts);
        }

        // read spatial coordinate arrays
        double bounds[6] = {0.0};
        unsigned long whole_extent[6] = {0ul};

        int x_id = 0;
        size_t n_x = 1;
        nc_type x_t = 0;
        teca_metadata x_atts;
        p_teca_variant_array x_axis;

        if (atrs.get(x_axis_variable, x_atts) ||
            x_atts.get("cf_dims", n_x) ||
            x_atts.get("cf_type_code", x_t) ||
            x_atts.get("cf_id", x_id))
        {
            this->clear_cached_metadata();
            TECA_FATAL_ERROR(
                << "Failed to get the attributes for x-axis variable \""
                << x_axis_variable << "\"")
            return teca_metadata();
        }

        NC_DISPATCH(x_t,
            size_t x_0 = 0;
            auto [x, px] = ::New<NC_TT>(n_x);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_get_vara(fh.get(), x_id, &x_0, &n_x, px)) != NC_NOERR)
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to read x axis, " << x_axis_variable << endl
                    << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            x_axis = x;
            whole_extent[1] = n_x - 1;
            bounds[0] = x->get(0);
            bounds[1] = x->get(whole_extent[1]);
            )

        // ensure that the x axis is not an integer type
        if ((x_t != NC_FLOAT) && (x_t != NC_DOUBLE))
        {
            TECA_WARNING("Integer x coordinate axes detected in " << x_axis_variable
                << ". A conversion to float will be made.")

            p_teca_float_array tmp_x = teca_float_array::New();
            tmp_x->assign(x_axis);
            x_axis = tmp_x;

            x_atts.set("type_code", x_axis->type_code());
            atrs.set(x_axis_variable, x_atts);

            x_t = NC_FLOAT;
        }

        int y_id = 0;
        size_t n_y = 1;
        nc_type y_t = 0;
        teca_metadata y_atts;
        p_teca_variant_array y_axis;
        if (!y_axis_variable.empty())
        {
            if (atrs.get(y_axis_variable, y_atts) ||
                y_atts.get("cf_dims", n_y) ||
                y_atts.get("cf_type_code", y_t) ||
                y_atts.get("cf_id", y_id))
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to get the attributes for y-axis variable \""
                    << y_axis_variable << "\"")
                return teca_metadata();
            }

            NC_DISPATCH(y_t,
                size_t y_0 = 0;
                auto [y, py] = ::New<NC_TT>(n_y);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(fh.get(), y_id, &y_0, &n_y, py)) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_FATAL_ERROR(
                        << "Failed to read y axis, " << y_axis_variable << endl
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                y_axis = y;
                whole_extent[3] = n_y - 1;
                bounds[2] = y->get(0);
                bounds[3] = y->get(whole_extent[3]);
                )

           // ensure consistent type accross coordinate axes.
           if (y_t != x_t)
           {
               TECA_WARNING("Mismatched y coordinate axes detected in " << y_axis_variable
                   << ". A type conversion will be made.")

               p_teca_variant_array tmp_y = x_axis->new_instance();
               tmp_y->assign(y_axis);
               y_axis = tmp_y;

               y_atts.set("type_code", y_axis->type_code());
               atrs.set(y_axis_variable, y_atts);

               y_t = x_t;
           }
        }
        else
        {
            NC_DISPATCH_FP(x_t,
                NC_PT y = NC_TT::New(1, NC_NT());
                y_axis = y;
                )
        }

        int z_id = 0;
        size_t n_z = 1;
        nc_type z_t = 0;
        teca_metadata z_atts;
        p_teca_variant_array z_axis;
        if (!z_axis_variable.empty())
        {
            if (atrs.get(z_axis_variable, z_atts) ||
                z_atts.get("cf_dims", n_z) ||
                z_atts.get("cf_type_code", z_t) ||
                z_atts.get("cf_id", z_id))
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to get the attributes for z-axis variable \""
                    << z_axis_variable << "\"")
                return teca_metadata();
            }

            NC_DISPATCH(z_t,
                size_t z_0 = 0;
                auto [z, pz] = ::New<NC_TT>(n_z);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(fh.get(), z_id, &z_0, &n_z, pz)) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_FATAL_ERROR(
                        << "Failed to read z axis, " << z_axis_variable << endl
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                z_axis = z;
                whole_extent[5] = n_z - 1;
                bounds[4] = z->get(0);
                bounds[5] = z->get(whole_extent[5]);
                )

           // ensure consistent type accross coordinate axes.
           if (z_t != x_t)
           {
               TECA_WARNING("Mismatched z coordinate axes detected in " << z_axis_variable
                   << ". A type conversion will be made.")

               p_teca_variant_array tmp_z = x_axis->new_instance();
               tmp_z->assign(z_axis);
               z_axis = tmp_z;

               z_atts.set("type_code", z_axis->type_code());
               atrs.set(z_axis_variable, z_atts);

               z_t = x_t;
           }
        }
        else
        {
            NC_DISPATCH(x_t,
                NC_PT z = NC_TT::New(1, NC_NT());
                z_axis = z;
                )
        }

        // we rely t_axis_variable being empty to indicate either that
        // there is no time axis, or that a time axis will be defined by
        // other algorithm properties. This temporary is used for metadata
        // consistency across those cases.
        std::string t_axis_var = t_axis_variable;

        p_teca_variant_array t_axis;
        teca_metadata t_atts;

        std::vector<unsigned long> step_count;
        if (!t_axis_variable.empty())
        {
            // validate the time axis calendaring metadata. this code is to
            // let us know when the time axis is not correctly specified in
            // the input file.
            teca_metadata time_atts;
            if (atrs.get(t_axis_variable, time_atts))
            {
                TECA_WARNING("Attribute metadata for time axis variable \""
                    << t_axis_variable << "\" is missing, Temporal analysis is "
                    << "likely to fail.")
            }

            // override the calendar
            if (!this->calendar.empty())
            {
                TECA_WARNING("Overriding the calendar with the runtime "
                    "provided value \"" << this->calendar << "\"")
                time_atts.set("calendar", this->calendar);
            }

            // override the units
            if (!this->t_units.empty())
            {
                TECA_WARNING("Overriding the time units with the runtime "
                    "provided value \"" << this->t_units << "\"")
                time_atts.set("units", this->t_units);
            }

            // check for units. units are necessary.
            int has_units = 0;
            if (!(has_units = time_atts.has("units")))
            {
                TECA_WARNING("The units attribute for the time axis variable \""
                    << t_axis_variable << "\" is missing. Temporal analysis is "
                    << "likely to fail.")
            }

            // check for calendar. calendar, if missing will be set to "standard"
            int has_calendar = 0;
            if (!(has_calendar = time_atts.has("calendar")))
            {
                TECA_WARNING("The calendar attribute for the time axis variable \""
                    << t_axis_variable << "\" is missing. Using the \"standard\" "
                    "calendar")
                time_atts.set("calendar", std::string("standard"));
            }

            // correct the data type if applying a user provided override
            if (!this->t_values.empty())
            {
                time_atts.set("cf_type_code",
                    int(teca_netcdf_util::netcdf_tt<double>::type_code));

                time_atts.set("type_code",
                    teca_variant_array_code<double>::get());
            }

            // get the base calendar and units. all the files are required to
            // use the same calendar, but in the case that some of the files
            // have different untis we will convert them into the base units.
            std::string base_calendar;
            time_atts.get("calendar", base_calendar);

            std::string base_units;
            time_atts.get("units", base_units);

            // save the updates
            atrs.set(t_axis_variable, time_atts);

            // process time axis
            const teca_cf_time_axis_data::elem_t &elem_0 = time_axis_data->get(0);
            const_p_teca_variant_array t0 = teca_cf_time_axis_data::get_variant_array(elem_0);
            if (!t0)
            {
                TECA_FATAL_ERROR("Failed to read time axis")
                return teca_metadata();
            }
            t_axis = t0->new_instance();

            for (size_t i = 0; i < n_files; ++i)
            {
                const teca_cf_time_axis_data::elem_t &elem_i = time_axis_data->get(i);

                // get the values read
                const_p_teca_variant_array t_i = teca_cf_time_axis_data::get_variant_array(elem_i);
                if (!t_i || !t_i->size())
                {
                    TECA_FATAL_ERROR("File " << i << " \"" << files[i]
                        << "\" had no time values")
                    return teca_metadata();
                }

                // it is an error for the files to have different calendars
                const teca_metadata &md_i = teca_cf_time_axis_data::get_metadata(elem_i);
                std::string calendar_i;
                md_i.get("calendar", calendar_i);
                if (this->calendar.empty() && (has_calendar || !calendar_i.empty())
                    && (calendar_i != base_calendar))
                {
                    TECA_FATAL_ERROR("The base calendar is \"" << base_calendar
                        << "\" but file " << i << " \"" << files[i]
                        << "\" has the \"" << calendar_i <<  "\" calendar")
                    return teca_metadata();
                }

                // we assume data is on the CPU
                assert(t_i->host_accessible());

                // update the step map
                size_t n_ti = t_i->size();
                step_count.push_back(n_ti);

                // allocate space to hold incoming values
                size_t n_t = t_axis->size();
                t_axis->resize(n_t + n_ti);

                std::string units_i;
                md_i.get("units", units_i);
                if (!this->t_units.empty() || (units_i == base_units))
                {
                    // the files are in the same units copy the data
                    VARIANT_ARRAY_DISPATCH(t_axis.get(),

                        auto [p_ti] = data<CTT>(t_i);
                        auto [p_t] = data<TT>(t_axis);

                        p_t += n_t;

                        memcpy(p_t, p_ti, sizeof(NT)*n_ti);
                        )
                }
                else
                {
                    // if there are no units present then we can not do a conversion
                    if (!has_units)
                    {
                        TECA_FATAL_ERROR("Calendaring conversion requires time units")
                        return teca_metadata();
                    }

                    // the files are in a different units, warn and convert
                    // to the base units
                    TECA_WARNING("File " << i << " \"" << files[i] << "\" units \""
                        << units_i << "\" differs from base units \"" << base_units
                        << "\" a conversion will be made.")

                    VARIANT_ARRAY_DISPATCH(t_axis.get(),

                        auto [p_ti] = data<CTT>(t_i);
                        auto [p_t] = data<TT>(t_axis);

                        p_t += n_t;

                        for (size_t j = 0; j < n_ti; ++j)
                        {
                            // convert offset from units_i to time
                            int YY=0;
                            int MM=0;
                            int DD=0;
                            int hh=0;
                            int mm=0;
                            double ss=0.0;
                            if (teca_calcalcs::date(double(p_ti[j]), &YY, &MM, &DD, &hh, &mm, &ss,
                                units_i.c_str(), base_calendar.c_str()))
                            {
                                TECA_FATAL_ERROR("Failed to convert offset ti[" << j << "] = "
                                    << p_ti[j] << " calendar \"" << base_calendar
                                    << "\" units \"" << units_i << "\" to time")
                                return teca_metadata();
                            }

                            // convert time to offsets from base units
                            double offs = 0.0;
                            if (teca_calcalcs::coordinate(YY, MM, DD, hh, mm, ss,
                                base_units.c_str(), base_calendar.c_str(), &offs))
                            {
                                TECA_FATAL_ERROR("Failed to convert time "
                                    << YY << "-" << MM << "-" << DD << " " << hh << ":"
                                    << mm << ":" << ss << " to offset in calendar \""
                                    << base_calendar << "\" units \"" << base_units
                                    << "\"")
                                return teca_metadata();
                            }

                            p_t[j] = offs;
#ifdef TECA_DEBUG
                            std::cerr
                                << YY << "-" << MM << "-" << DD << " " << hh << ":"
                                << mm << ":" << ss << " "  << p_ti[j] << " -> " << offs
                                << std::endl;
#endif
                        }
                        )
                }
            }

            // override the time values read from disk with user supplied set
            if (!this->t_values.empty())
            {

                TECA_WARNING("Overriding the time coordinates stored on disk "
                    "with runtime provided values.")

                size_t n_t_vals = this->t_values.size();
                if (n_t_vals != t_axis->size())
                {
                    TECA_FATAL_ERROR("Number of timesteps detected doesn't match "
                        "the number of time values provided; " << n_t_vals
                        << " given, " << t_axis->size() << " are necessary.")
                    return teca_metadata();
                }

                p_teca_double_array t = teca_double_array::New
                    (n_t_vals, this->t_values.data());

                t_axis = t;
            }
        }
        else if (!this->t_values.empty())
        {
            TECA_STATUS("The t_axis_variable was unspecified, using the "
                "provided time values")

            if (this->calendar.empty() || this->t_units.empty())
            {
                TECA_FATAL_ERROR("The calendar and units must to be specified when "
                    " providing time values")
                return teca_metadata();
            }

            // if time axis is provided manually by the user
            size_t n_t_vals = this->t_values.size();
            if (n_t_vals != files.size())
            {
                TECA_FATAL_ERROR("Number of files choosen doesn't match the"
                    " number of time values provided; " << n_t_vals <<
                    " given, " << files.size() << " detected.")
                return teca_metadata();
            }

            teca_metadata time_atts;
            time_atts.set("calendar", this->calendar);
            time_atts.set("units", this->t_units);
            time_atts.set("cf_dims", n_t_vals);
            time_atts.set("cf_type_code", int(teca_netcdf_util::netcdf_tt<double>::type_code));
            time_atts.set("type_code", teca_variant_array_code<double>::get());
            time_atts.set("centering", int(teca_array_attributes::point_centering));

            atrs.set("time", time_atts);

            p_teca_double_array t =
                teca_double_array::New
                    (n_t_vals, this->t_values.data());

            step_count.resize(n_t_vals, 1);

            t_axis = t;

            t_axis_var = "time";
        }
        // infer the time from the filenames
        else if (!this->filename_time_template.empty())
        {
            std::vector<double> t_values;

            std::string t_units = this->t_units;
            std::string calendar = this->calendar;

            // assume that this is a standard calendar if none is provided
            if (this->calendar.empty())
            {
                calendar = "standard";
            }

            // loop over all files and infer dates from names
            size_t n_files = files.size();
            for (size_t i = 0; i < n_files; ++i)
            {
                struct tm current_tm;
                memset(&current_tm, 0, sizeof(current_tm));

                if (!strptime(files[i].c_str(), this->filename_time_template.c_str(), &current_tm))
                {
                    TECA_FATAL_ERROR("Failed to infer time from filename \"" <<
                        files[i] << "\" using format \"" <<
                        this->filename_time_template << "\"")
                    return teca_metadata();
                }

                // set the time units based on the first file date if we
                // don't have time units
                if ((i == 0) && t_units.empty())
                {
                    std::string t_units_fmt =
                        "days since %Y-%m-%d 00:00:00";

                    // convert the time data to a string
                    char tmp[256];
                    if (strftime(tmp, sizeof(tmp), t_units_fmt.c_str(),
                          &current_tm) == 0)
                    {
                        TECA_FATAL_ERROR(
                            "failed to convert the time as a string with \""
                            << t_units_fmt << "\"")
                        return teca_metadata();
                    }
                    // save the time units
                    t_units = tmp;
                }
#if defined(TECA_HAS_UDUNITS)
                // convert the time to a double using calcalcs
                int year = current_tm.tm_year + 1900;
                int mon = current_tm.tm_mon + 1;
                int day = current_tm.tm_mday;
                int hour = current_tm.tm_hour;
                int minute = current_tm.tm_min;
                double second = current_tm.tm_sec;
                double current_time = 0;
                if (teca_calcalcs::coordinate(year, mon, day, hour, minute,
                    second, t_units.c_str(), calendar.c_str(), &current_time))
                {
                    TECA_FATAL_ERROR("conversion of date inferred from "
                        "filename failed");
                    return teca_metadata();
                }
                // add the current time to the list
                t_values.push_back(current_time);
#else
                TECA_FATAL_ERROR("The UDUnits package is required for this operation")
                return teca_metadata();
#endif
            }

            TECA_STATUS("The time axis will be infered from file names using "
                "the user provided template \"" << this->filename_time_template
                << "\" with the \"" << calendar << "\" calendar in units \""
                << t_units << "\"")

            // create a teca variant array from the times
            size_t n_t_vals = t_values.size();
            p_teca_double_array t =
                teca_double_array::New
                    (n_t_vals, t_values.data());

            // set the number of time steps
            step_count.resize(n_t_vals, 1);

            // set the time metadata
            teca_metadata time_atts;
            time_atts.set("calendar", calendar);
            time_atts.set("units", t_units);
            time_atts.set("cf_dims", n_t_vals);
            time_atts.set("cf_type_code", int(teca_netcdf_util::netcdf_tt<double>::type_code));
            time_atts.set("type_code", teca_variant_array_code<double>::get());
            time_atts.set("centering", int(teca_array_attributes::point_centering));
            atrs.set("time", time_atts);

            // set the time axis
            t_axis = t;
            t_axis_var = "time";
        }
        else
        {
            // make a dummy time axis, this enables parallelization over
            // file sets that do not have time dimension. However, there is
            // no guarantee on the order of the dummy axis to the lexical
            // ordering of the files and there will be no calendaring
            // information. As a result many time aware algorithms will not
            // work.
            size_t n_files = files.size();
            NC_DISPATCH(x_t,
                auto [t, pt] = ::New<NC_TT>(n_files);
                for (size_t i = 0; i < n_files; ++i)
                {
                    pt[i] = NC_NT(i);
                    step_count.push_back(1);
                }
                t_axis = t;
                )

            t_axis_var = "time";

            TECA_STATUS("The time axis will be generated, with 1 step per file")
        }

        // convert time axis to floating point
        if (!std::dynamic_pointer_cast<teca_double_array>(t_axis) &&
            !std::dynamic_pointer_cast<teca_float_array>(t_axis))
        {
            p_teca_float_array t = teca_float_array::New();
            t->assign(t_axis);
            t_axis = t;
        }

        this->internals->metadata.set("variables", vars);
        this->internals->metadata.set("attributes", atrs);

        teca_metadata coords;
        coords.set("x_variable", x_axis_variable);
        coords.set("y_variable", (y_axis_variable.empty() ? "y" : y_axis_variable));
        coords.set("z_variable", (z_axis_variable.empty() ? "z" : z_axis_variable));
        coords.set("t_variable", t_axis_var);
        coords.set("x", x_axis);
        coords.set("y", y_axis);
        coords.set("z", z_axis);
        coords.set("t", t_axis);
        coords.set("periodic_in_x", this->periodic_in_x);
        coords.set("periodic_in_y", this->periodic_in_y);
        coords.set("periodic_in_z", this->periodic_in_z);
        this->internals->metadata.set("whole_extent", whole_extent);
        this->internals->metadata.set("bounds", bounds);
        this->internals->metadata.set("coordinates", coords);
        this->internals->metadata.set("files", files);
        this->internals->metadata.set("root", path);
        this->internals->metadata.set("step_count", step_count);
        this->internals->metadata.set("number_of_time_steps",
                t_axis->size());

        // inform the executive how many and how to request time steps
        this->internals->metadata.set(
            "index_initializer_key", std::string("number_of_time_steps"));

        this->internals->metadata.set(
            "index_request_key", std::string("temporal_extent"));

        this->internals->metadata.to_stream(stream);

#if defined(TECA_HAS_MPI)
        // broadcast the metadata to other ranks
        if (is_init)
            stream.broadcast(comm, root_rank);
#endif
    }
#if defined(TECA_HAS_MPI)
    else
    if (is_init)
    {
        // all other ranks receive the metadata from the root
        stream.broadcast(comm, root_rank);

        this->internals->metadata.from_stream(stream);

        // initialize the file map
        std::vector<std::string> files;
        this->internals->metadata.get("files", files);
    }
#endif

    return this->internals->metadata;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cf_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_reader::execute" << endl;
#endif
    (void)port;
    (void)input_data;

    int rank = 0;
#if defined(TECA_HAS_MPI)
    MPI_Comm comm = this->get_communicator();

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
    }
#endif

    // get coordinates
    teca_metadata coords;
    if (this->internals->metadata.get("coordinates", coords))
    {
        TECA_FATAL_ERROR("metadata is missing \"coordinates\"")
        return nullptr;
    }

    p_teca_variant_array in_x, in_y, in_z, in_t;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")) || !(in_t = coords.get("t")))
    {
        TECA_FATAL_ERROR("metadata is missing coordinate arrays")
        return nullptr;
    }

    // assume coordinate data is on the CPU
    assert(in_x->host_accessible());
    assert(in_y->host_accessible());
    assert(in_z->host_accessible());

    // the requested extents must not exceed these limits
    unsigned long nx_max = in_x->size();
    unsigned long ny_max = in_y->size();
    unsigned long nz_max = in_z->size();

    // get names, need to be careful since some of these depend
    // on run time information. eg: user can specify a time axis
    // via algorithm properties
    std::string x_axis_var;
    std::string y_axis_var;
    std::string z_axis_var;
    std::string t_axis_var;
    coords.get("x_variable", x_axis_var);
    coords.get("y_variable", y_axis_var);
    coords.get("z_variable", z_axis_var);
    coords.get("t_variable", t_axis_var);

    // get requested time extent
    unsigned long temporal_extent[2] = {0ul};
    double temporal_bounds[2] = {0.0};
    if (!request.get("time", temporal_bounds[0]))
    {
        // translate time to a time step
        VARIANT_ARRAY_DISPATCH_FP(in_t.get(),
            auto [pin_t] = data<TT>(in_t);
            if (teca_coordinate_util::index_of(pin_t, 0,
                in_t->size()-1, static_cast<NT>(temporal_bounds[0]), temporal_extent[0]))
            {
                TECA_FATAL_ERROR("requested time " << temporal_bounds[0] << " not found")
                return nullptr;
            }
            )
        temporal_extent[1] = temporal_extent[0];
        temporal_bounds[1] = temporal_bounds[0];
    }
    else if (!request.get("temporal_bounds", temporal_bounds))
    {
        // translate time range to a time step range
        if (teca_coordinate_util::bounds_to_extent(temporal_bounds, in_t, temporal_extent))
        {
            TECA_FATAL_ERROR("The requested temporal_bounds ["
                << temporal_bounds << "] was not found")
            return nullptr;
        }
    }
    else
    {
        // get the step number or step range directly

        // note: there is no error checking here to support case where only 1
        // time step is present in a dataset. In that case the single step is
        // always served regardless of the request (or lack there of).
        unsigned long n_steps = in_t->size();

        request.get("temporal_extent", temporal_extent);

        if (in_t)
        {
            if (in_t->size() == 1)
            {
                temporal_extent[0] = temporal_extent[1] = 0;
            }

            if (!((temporal_extent[0] < n_steps) && (temporal_extent[1] < n_steps)))
            {
                TECA_FATAL_ERROR("Invalid time extent [" << temporal_extent
                    << "] requested from data set with " << in_t->size()
                    << " steps")
                return nullptr;
            }

            in_t->get(temporal_extent[0], temporal_bounds[0]);
            in_t->get(temporal_extent[1], temporal_bounds[1]);
        }
    }

    unsigned long whole_extent[6] = {0};
    if (this->internals->metadata.get("whole_extent", whole_extent, 6))
    {
        TECA_FATAL_ERROR("temporal_extent = [" << temporal_extent
            << "] metadata is missing \"whole_extent\"")
        return nullptr;
    }

    unsigned long extent[6] = {0};
    double bounds[6] = {0.0};
    if (request.get("bounds", bounds, 6))
    {
        // bounds key not present, check for extent key
        // if not present use whole_extent
        if (request.get("extent", extent, 6))
        {
            memcpy(extent, whole_extent, 6*sizeof(unsigned long));
        }
        else
        {
            // clamp the extent. this is a workaround to support 2D data
            // with a vertical dimension of 1 being read into a 3D mesh.
            if (this->clamp_dimensions_of_one)
            {
                teca_coordinate_util::clamp_dimensions_of_one(
                    nx_max, ny_max, nz_max, extent, this->verbose);
            }

            // validate the requested extent before using it
            if (teca_coordinate_util::validate_extent(nx_max,
                ny_max, nz_max, extent, true))
            {
                TECA_FATAL_ERROR("An invalid extent [" << extent
                    << "] was requested. The available extent is [0, "
                    << nx_max << ", 0, " << ny_max << ", 0, " << nz_max
                    << "]")
                return nullptr;
            }
        }

        // get bounds of the extent being read
        in_x->get(extent[0], bounds[0]);
        in_x->get(extent[1], bounds[1]);
        in_y->get(extent[2], bounds[2]);
        in_y->get(extent[3], bounds[3]);
        in_z->get(extent[4], bounds[4]);
        in_z->get(extent[5], bounds[5]);
    }
    else
    {
        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        if (teca_coordinate_util::bounds_to_extent(
            bounds, in_x, in_y, in_z, extent))
        {
            TECA_FATAL_ERROR("invalid bounds requested.")
            return nullptr;
        }

        // clamp the extent. this is a workaround to support 2D data
        // with a vertical dimension of 1 being read into a 3D mesh.
        if (this->clamp_dimensions_of_one)
        {
            teca_coordinate_util::clamp_dimensions_of_one(
                nx_max, ny_max, nz_max, extent, this->verbose);
        }

        // validate the requested extent before using it
        if (teca_coordinate_util::validate_extent(nx_max,
            ny_max, nz_max, extent, true))
        {
            TECA_FATAL_ERROR("An invalid extent [" << extent
                << "] was requested. The available extent is [0, "
                << nx_max << ", 0, " << ny_max << ", 0, " << nz_max
                << "]")
            return nullptr;
        }
    }

    // requesting arrays is optional, but it's an error
    // to request an array that isn't present.
    std::vector<std::string> arrays;
    request.get("arrays", arrays);
    size_t n_arrays = arrays.size();

    // slice axes on the requested extent
    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    p_teca_variant_array out_x = in_x->new_copy(extent[0], nx);
    p_teca_variant_array out_y = in_y->new_copy(extent[2], ny);
    p_teca_variant_array out_z = in_z->new_copy(extent[4], nz);

    // locate files with these time steps
    std::vector<unsigned long> step_count;
    if (this->internals->metadata.get("step_count", step_count))
    {
        TECA_FATAL_ERROR("temporal_extent = [" << temporal_extent
            << "] metadata is missing \"step_count\"")
        return nullptr;
    }

    // compute the partial sum of the step counts.
    unsigned long n_files = step_count.size();
    std::vector<unsigned long> step_count_sum(n_files);
    unsigned long ps = 0;
    for (size_t i = 0; i < n_files; ++i)
    {
        ps += step_count[i];
        step_count_sum[i] = ps - 1;
    }

    unsigned long i = 0;
    unsigned long file_index[2] = {0ul};

    while (step_count_sum[i] < temporal_extent[0])
    {
        ++i;

        if (i >= n_files)
        {
            TECA_FATAL_ERROR("file not found for the requested step "
                << temporal_extent[0])
            return nullptr;
        }
    }

    file_index[0] = i;

    while (step_count_sum[i] < temporal_extent[1])
    {
        ++i;

        if (i >= n_files)
        {
            TECA_FATAL_ERROR("file not found for the requested step "
                << temporal_extent[1])
            return nullptr;
        }
    }

    file_index[1] = i;

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();

    mesh->set_index_request_key("temporal_extent");
    mesh->set_x_coordinates(x_axis_var, out_x);
    mesh->set_y_coordinates(y_axis_var, out_y);
    mesh->set_z_coordinates(z_axis_var, out_z);
    mesh->set_temporal_bounds(temporal_bounds);
    mesh->set_temporal_extent(temporal_extent);
    mesh->set_whole_extent(whole_extent);
    mesh->set_extent(extent);
    mesh->set_bounds(bounds);
    mesh->set_periodic_in_x(this->periodic_in_x);
    mesh->set_periodic_in_y(this->periodic_in_y);
    mesh->set_periodic_in_z(this->periodic_in_z);

    // get the array attributes
    teca_metadata atrs;
    if (this->internals->metadata.get("attributes", atrs))
    {
        TECA_FATAL_ERROR("Dataset metadata is missing \"attributes\"")
        return nullptr;
    }

    // pass time axis attributes
    teca_metadata time_atts;
    std::string calendar;
    std::string units;
    if (!atrs.get(t_axis_var, time_atts)
       && !time_atts.get("calendar", calendar)
       && !time_atts.get("units", units))
    {
        mesh->set_calendar(calendar);
        mesh->set_time_units(units);
    }

    // pass the attributes for the arrays read
    teca_metadata out_atrs;
    for (unsigned int i = 0; i < n_arrays; ++i)
        out_atrs.set(arrays[i], atrs.get(arrays[i]));

    // pass coordinate axes attributes
    if (atrs.has(x_axis_var))
        out_atrs.set(x_axis_var, atrs.get(x_axis_var));
    if (atrs.has(y_axis_var))
        out_atrs.set(y_axis_var, atrs.get(y_axis_var));
    if (atrs.has(z_axis_var))
        out_atrs.set(z_axis_var, atrs.get(z_axis_var));
    if (!time_atts.empty())
        out_atrs.set(t_axis_var, time_atts);

    mesh->set_attributes(out_atrs);

    // get the requested target device
    allocator alloc = allocator::malloc;
    allocator tmp_alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        // place all data read on the assigned device
        teca_cuda_util::set_device(device_id);
        alloc = allocator::cuda_async;
        // using the cuda host allocator slowed things down on Perlmutter
        //tmp_alloc = allocator::cuda_host;
    }
#endif

    // allocate space for the array. this is done in prior to the reads because
    // when reading multiple time steps we may need to read from multiple files
    // and we want to pre-allocate space large enough to hold the entire result
    size_t n_time_steps = temporal_extent[1] - temporal_extent[0] + 1;
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get metadata
        teca_metadata atts;
        int type = 0;
        int id = 0;
        int have_mesh_dim[4] = {0};
        int mesh_dim_active[4] = {0};
        unsigned int centering = teca_array_attributes::no_centering;
        std::vector<size_t> cf_dims;

        if (atrs.get(arrays[i], atts)
            || atts.get("cf_type_code", 0, type)
            || atts.get("cf_id", 0, id)
            || atts.get("cf_dims", cf_dims)
            || atts.get("centering", centering)
            || atts.get("have_mesh_dim", have_mesh_dim, 4)
            || atts.get("mesh_dim_active", mesh_dim_active, 4))
        {
            TECA_FATAL_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
            continue;
        }

        size_t n_vals = 1;
        unsigned int n_dims = cf_dims.size();

        if (centering == teca_array_attributes::point_centering)
        {
            // select the requested time step
            // subset point centered variables based on the incoming requested
            // extent.
            if (have_mesh_dim[3])
            {
                n_vals *= n_time_steps;
            }

            if (have_mesh_dim[2])
            {
                n_vals *= mesh_dim_active[2] ? extent[5] - extent[4] + 1 : 1;
            }

            if (have_mesh_dim[1])
            {
                n_vals *= mesh_dim_active[1] ? extent[3] - extent[2] + 1 : 1;
            }

            if (have_mesh_dim[0])
            {
                n_vals *= mesh_dim_active[0] ? extent[1] - extent[0] + 1 : 1;
            }
        }
        else if (centering == teca_array_attributes::no_centering)
        {
            // most of the time this is a user error, forgetting to set z_axis_variable.
            if (have_mesh_dim[0] || have_mesh_dim[1] || have_mesh_dim[2])
            {
                TECA_WARNING("The variable \"" << arrays[i] << "\" is being treated"
                    " as an information array. It has dimensions matching those of the mesh"
                    " in the " << (have_mesh_dim[0] ? "x " : "") << (have_mesh_dim[1] ? "y " : "")
                    << (have_mesh_dim[2] ? "z " : "") << (have_mesh_dim[3] ? "t " : "")
                    << "directions but the " << (have_mesh_dim[0] ? "" : "x ")
                    << (have_mesh_dim[1] ? "" : "y ") << (have_mesh_dim[2] ? "" : "z ")
                    << (have_mesh_dim[3] ? "" : "t ") << " dimensions of the mesh have been"
                    " disabled.")
            }

            // read non-spatial data
            // if the first dimension is time then select the requested time
            // step. otherwise read the entire thing
            if (!t_axis_variable.empty() && have_mesh_dim[3])
            {
                n_vals *= n_time_steps;
            }
            else
            {
                n_vals *= cf_dims[0];
            }

            for (unsigned int ii = 1; ii < n_dims; ++ii)
            {
                n_vals *= cf_dims[ii];
            }
        }
        else
        {
            TECA_FATAL_ERROR("Invalid centering can't read \"" << arrays[i] << "\"")
            continue;
        }

        // allocate the array and reserve space, later we append data from each
        // file. save the array in the output mesh.
        NC_DISPATCH(type,
            NC_PT a = NC_TT::New(alloc);
            a->reserve(n_vals);
            mesh->get_arrays(centering)->append(arrays[i], a);
            )
    }

    // get the base path and list of files
    std::string path;
    p_teca_string_array files;
    if (this->internals->metadata.get("root", path)
        || !(files = std::dynamic_pointer_cast<teca_string_array>(this->internals->metadata.get("files"))))
    {
        TECA_FATAL_ERROR("Metadata is missing files")
        return nullptr;
    }

    // loop over the list of files to read the requested time range
    for (unsigned int j = file_index[0]; j <= file_index[1]; ++j)
    {
        // assume we'll read all steps in the file, correct for first and last
        // file in the series
        size_t first_step = 0;
        size_t num_steps = step_count[j];

        // correct for the first and last files
        if (j == file_index[0])
        {
            first_step = temporal_extent[0] - (j ? step_count_sum[j - 1] + 1 : 0);
        }

        if (j == file_index[1])
        {
            size_t last_step = temporal_extent[1] - (j ? step_count_sum[j - 1] + 1 : 0);
            num_steps = last_step - first_step + 1;
        }

        // get the file
        std::string file = files->get<std::string>(j);

        // open the handle for this file
        int ierr = 0;
        std::string file_path = path + PATH_SEP + file;
        teca_netcdf_util::netcdf_handle fh;

        if ((this->collective_buffer > 0) && is_init)
            ierr = fh.open(comm, file_path, NC_NOWRITE);
        else
            ierr = fh.open(file_path, NC_NOWRITE);

        if (ierr)
        {
            TECA_FATAL_ERROR("Failed to open \"" << file << "\"")
            return nullptr;
        }

        int file_id = fh.get();

        // read requested arrays
        for (size_t i = 0; i < n_arrays; ++i)
        {
            // get metadata
            teca_metadata atts;
            int type = 0;
            int id = 0;
            int have_mesh_dim[4] = {0};
            int mesh_dim_active[4] = {0};
            unsigned int centering = teca_array_attributes::no_centering;
            std::vector<size_t> cf_dims;

            if (atrs.get(arrays[i], atts)
                || atts.get("cf_type_code", 0, type)
                || atts.get("cf_id", 0, id)
                || atts.get("cf_dims", cf_dims)
                || atts.get("centering", centering)
                || atts.get("have_mesh_dim", have_mesh_dim, 4)
                || atts.get("mesh_dim_active", mesh_dim_active, 4))
            {
                TECA_FATAL_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
                continue;
            }

            size_t n_vals = 1;
            unsigned int n_dims = cf_dims.size();
            std::vector<size_t> starts;
            std::vector<size_t> counts;

            if (centering == teca_array_attributes::point_centering)
            {
#if defined(TECA_HAS_NETCDF_MPI)
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                // enable collective buffering. the writer uses -1 for
                // automatic, but the reader can't determine when it's safe, so
                // the setting must be explicit
                if ((this->collective_buffer > 0) && is_init)
                {
                    if ((ierr = nc_var_par_access(file_id, id, NC_COLLECTIVE)) != NC_NOERR)
                    {
                        if (rank == 0)
                        {
                            TECA_ERROR("Failed to set collective access mode"
                                " on variable \"" << arrays[i] << "\"" << std::endl
                                << nc_strerror(ierr))
                        }
                    }
                    else if (this->verbose && (rank == 0))
                    {
                        TECA_STATUS("Enabled collective read on \"" << arrays[i]  << "\"")
                    }
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
#endif
                // select the requested time step
                // subset point centered variables based on the incoming requested
                // extent.
                if (have_mesh_dim[3])
                {
                    starts.push_back(mesh_dim_active[3] ? first_step : 0);
                    counts.push_back(num_steps);

                    n_vals *= num_steps;
                }

                if (have_mesh_dim[2])
                {
                    size_t start = mesh_dim_active[2] ? extent[4] : 0;
                    size_t count = mesh_dim_active[2] ? extent[5] - extent[4] + 1 : 1;

                    starts.push_back(start);
                    counts.push_back(count);

                    n_vals *= count;
                }

                if (have_mesh_dim[1])
                {
                    size_t start = mesh_dim_active[1] ? extent[2] : 0;
                    size_t count = mesh_dim_active[1] ? extent[3] - extent[2] + 1 : 1;

                    starts.push_back(start);
                    counts.push_back(count);

                    n_vals *= count;
                }

                if (have_mesh_dim[0])
                {
                    size_t start = mesh_dim_active[0] ? extent[0] : 0;
                    size_t count = mesh_dim_active[0] ? extent[1] - extent[0] + 1 : 1;

                    starts.push_back(start);
                    counts.push_back(count);

                    n_vals *= count;
                }
            }
            else if (centering == teca_array_attributes::no_centering)
            {
                // most of the time this is a user error, forgetting to set z_axis_variable.
                if (have_mesh_dim[0] || have_mesh_dim[1] || have_mesh_dim[2])
                {
                    TECA_WARNING("The variable \"" << arrays[i] << "\" is being treated"
                        " as an information array. It has dimensions matching those of the mesh"
                        " in the " << (have_mesh_dim[0] ? "x " : "") << (have_mesh_dim[1] ? "y " : "")
                        << (have_mesh_dim[2] ? "z " : "") << (have_mesh_dim[3] ? "t " : "")
                        << "directions but the " << (have_mesh_dim[0] ? "" : "x ")
                        << (have_mesh_dim[1] ? "" : "y ") << (have_mesh_dim[2] ? "" : "z ")
                        << (have_mesh_dim[3] ? "" : "t ") << " dimensions of the mesh have been"
                        " disabled.")
                }

                // read non-spatial data
                // if the first dimension is time then select the requested time
                // step. otherwise read the entire thing
                if (!t_axis_variable.empty() && have_mesh_dim[3])
                {
                    starts.push_back(first_step);
                    counts.push_back(num_steps);

                    n_vals *= num_steps;
                }
                else
                {
                    starts.push_back(0);

                    size_t dim_len = cf_dims[0];
                    counts.push_back(dim_len);

                    n_vals = dim_len;
                }

                for (unsigned int ii = 1; ii < n_dims; ++ii)
                {
                    starts.push_back(0);

                    size_t dim_len = cf_dims[ii];
                    counts.push_back(dim_len);

                    n_vals *= dim_len;
                }
            }
            else
            {
                TECA_FATAL_ERROR("Invalid centering can't read \"" << arrays[i] << "\"")
                continue;
            }

            NC_DISPATCH(type,
                // read the data into a temprary array
                auto [a, pa] = ::New<NC_TT>(n_vals, tmp_alloc);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], pa)) != NC_NOERR)
                {
                    TECA_FATAL_ERROR("Failed to read variable \"" << arrays[i]
                        << "\" starts = [" << starts << "] counts = [" << counts
                        << "] " << file << endl << nc_strerror(ierr))
                    continue;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                // get the output array
                p_teca_variant_array array = mesh->get_arrays(centering)->get(arrays[i]);

                // append the temporary to the output
                static_cast<NC_TT*>(array.get())->append(a);
                )
        }
    }

    return mesh;
}
