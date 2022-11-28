#include "teca_wrf_reader.h"
#include "teca_array_attributes.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_file_util.h"
#include "teca_arakawa_c_grid.h"
#include "teca_coordinate_util.h"
#include "teca_netcdf_util.h"
#include "teca_calcalcs.h"

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

using std::endl;
using std::cerr;
using namespace teca_variant_array_util;

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_OPENSSL)
#include <openssl/sha.h>
#endif

// internals for the cf reader
class teca_wrf_reader_internals
{
public:
    teca_wrf_reader_internals()
    {}

#if defined(TECA_HAS_OPENSSL)
    // create a key used to identify metadata
    std::string create_metadata_cache_key(const std::string &path,
        const std::vector<std::string> &files);
#endif

public:
    teca_metadata metadata;
};

#if defined(TECA_HAS_OPENSSL)
// --------------------------------------------------------------------------
std::string teca_wrf_reader_internals::create_metadata_cache_key(
    const std::string &path, const std::vector<std::string> &files)
{
    // create the hash using the version, file names, and path
    SHA_CTX ctx;
    SHA1_Init(&ctx);

    // include the version since metadata could change between releases
    SHA1_Update(&ctx, TECA_VERSION_DESCR, strlen(TECA_VERSION_DESCR));

    // include path to the data
    SHA1_Update(&ctx, path.c_str(), path.size());

    // include each file. different regex could identify different sets
    // of files.
    unsigned long n = files.size();
    for (unsigned long i = 0; i < n; ++i)
    {
        const std::string &file_name = files[i];
        SHA1_Update(&ctx, file_name.c_str(), file_name.size());
    }

    unsigned char key[SHA_DIGEST_LENGTH] = {0};
    SHA1_Final(key, &ctx);

    // convert to ascii
    std::ostringstream oss;
    oss.fill('0');
    oss << std::hex;

    for (unsigned int i = 0; i < SHA_DIGEST_LENGTH; ++i)
        oss << std::setw(2) << static_cast<unsigned int>(key[i]);

    return oss.str();
}
#endif


// --------------------------------------------------------------------------
teca_wrf_reader::teca_wrf_reader() :
    files_regex(""),
    m_x_axis_variable("XLONG"),
    m_y_axis_variable("XLAT"),
    u_x_axis_variable("XLONG_U"),
    u_y_axis_variable("XLAT_U"),
    v_x_axis_variable("XLONG_V"),
    v_y_axis_variable("XLAT_V"),
    m_z_axis_variable("ZNU"),
    w_z_axis_variable("ZNW"),
    t_axis_variable("XTIME"),
    calendar(""),
    t_units(""),
    filename_time_template(""),
    periodic_in_x(0),
    periodic_in_y(0),
    periodic_in_z(0),
    thread_pool_size(-1),
    internals(new teca_wrf_reader_internals)
{}

// --------------------------------------------------------------------------
teca_wrf_reader::~teca_wrf_reader()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_wrf_reader::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_wrf_reader":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, file_names,
            "paths/file names to read")
        TECA_POPTS_GET(std::string, prefix, files_regex,
            "a regular expression that matches the set of files "
            "comprising the dataset")
        TECA_POPTS_GET(std::string, prefix, metadata_cache_dir,
            "a directory where metadata caches can be stored")
        TECA_POPTS_GET(std::string, prefix, calendar,
            "name of variable that has the time calendar")
        TECA_POPTS_GET(std::string, prefix, t_units,
            "a std::get_time template for decoding time from the input filename")
        TECA_POPTS_GET(std::string, prefix, filename_time_template,
            "name of variable that has the time unit")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, t_values,
            "name of variable that has t axis values set by the"
            "the user if the file doesn't have time variable set")
        TECA_POPTS_GET(int, prefix, periodic_in_x,
            "the dataset has apriodic boundary in the x direction")
        TECA_POPTS_GET(int, prefix, periodic_in_y,
            "the dataset has apriodic boundary in the y direction")
        TECA_POPTS_GET(int, prefix, periodic_in_z,
            "the dataset has apriodic boundary in the z direction")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "set the number of I/O threads")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_wrf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, file_names)
    TECA_POPTS_SET(opts, std::string, prefix, files_regex)
    TECA_POPTS_SET(opts, std::string, prefix, metadata_cache_dir)
    TECA_POPTS_SET(opts, std::string, prefix, calendar)
    TECA_POPTS_SET(opts, std::string, prefix, t_units)
    TECA_POPTS_SET(opts, std::string, prefix, filename_time_template)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, t_values)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_x)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_y)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_z)
    TECA_POPTS_SET(opts, int, prefix, thread_pool_size)
}
#endif

// --------------------------------------------------------------------------
void teca_wrf_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_wrf_reader::clear_cached_metadata()
{
    this->internals->metadata.clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_wrf_reader::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_wrf_reader::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    // return cached metadata. cache is cleared if
    // any of the algorithms properties are modified
    if (this->internals->metadata)
        return this->internals->metadata;

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
    teca_binary_stream stream;

    // only rank 0 will parse the dataset. once
    // parsed metadata is broadcast to all
    int root_rank = n_ranks - 1;
    if (rank == root_rank)
    {
        std::vector<std::string> files;
        std::string path;

        if (!this->file_names.empty())
        {
            // use file name
            size_t file_names_size = this->file_names.size();
            for (size_t i = 0; i < file_names_size; ++i)
            {
                std::string file_name = this->file_names[i];
                path = teca_file_util::path(file_name);
                files.push_back(teca_file_util::filename(file_name));
            }
        }
        else
        {
            // use regex
            std::string regex = teca_file_util::filename(this->files_regex);
            path = teca_file_util::path(this->files_regex);

            if (teca_file_util::locate_files(path, regex, files))
            {
                TECA_FATAL_ERROR(
                    << "Failed to locate any files" << endl
                    << this->files_regex << endl
                    << path << endl
                    << regex)
                return teca_metadata();
            }
        }

#if defined(TECA_HAS_OPENSSL)
        // look for a metadata cache. we are caching it on disk as for large
        // datasets on Lustre, scanning the time dimension is costly because of
        // NetCDF CF convention that time is unlimitted and thus not layed out
        // contiguously in the files.
        std::string metadata_cache_key =
            this->internals->create_metadata_cache_key(path, files);

        std::string metadata_cache_path[4] =
            {(getenv("HOME") ? : "."), ".", path, metadata_cache_dir};

        int n_metadata_cache_paths = metadata_cache_dir.empty() ? 2 : 3;

        for (int i = n_metadata_cache_paths; i >= 0; --i)
        {
            std::string metadata_cache_file =
                metadata_cache_path[i] + PATH_SEP + "." + metadata_cache_key + ".tmd";

            if (teca_file_util::file_exists(metadata_cache_file.c_str()))
            {
                // read the cache
                if (teca_file_util::read_stream(metadata_cache_file.c_str(),
                    "teca_wrf_reader::metadata_cache_file", stream))
                {
                    TECA_WARNING("Failed to read metadata cache \""
                        << metadata_cache_file << "\"")
                }
                else
                {
                    TECA_STATUS("Found metadata cache \""
                        << metadata_cache_file << "\"")
                    // recover metadata
                    this->internals->metadata.from_stream(stream);
                    // stop
                    break;
                }
            }
        }
#endif

        // load from cache failed, generate from scratch
        if (!this->internals->metadata)
        {
            int ierr = 0;
            std::string file = path + PATH_SEP + files[0];

            // get mesh coordinates and dimensions
            int x_id = 0;
            int z_id = 0;
            int x_ndims = 0;
            int z_ndims = 0;
            int x_dims[NC_MAX_VAR_DIMS] = {0};
            int z_dims[NC_MAX_VAR_DIMS] = {0};
            size_t n_x = 1;
            size_t n_y = 1;
            size_t n_z = 1;
            nc_type x_t = 0;
            nc_type z_t = 0;
            int n_vars = 0;
            p_teca_variant_array t_axis;
            //double bounds[6] = {0.0};
            unsigned long whole_extent[6] = {0ul};

            teca_metadata atrs;
            std::vector<std::string> vars;

            teca_netcdf_util::netcdf_handle fh;
            if (fh.open(file.c_str(), NC_NOWRITE))
            {
                TECA_FATAL_ERROR("Failed to open " << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }

            // query mesh axes
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if (((ierr = nc_inq_varid(fh.get(), m_x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                || ((ierr = nc_inq_varndims(fh.get(), x_id, &x_ndims)) != NC_NOERR)
                || ((ierr = nc_inq_vardimid(fh.get(), x_id, x_dims)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(fh.get(), x_id, &x_t)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), x_dims[2], &n_x)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), x_dims[1], &n_y)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to query x axis variable \"" << m_x_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
           }

#if !defined(HDF5_THREAD_SAFE)
            }
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if (((ierr = nc_inq_varid(fh.get(), m_z_axis_variable.c_str(), &z_id)) != NC_NOERR)
                || ((ierr = nc_inq_varndims(fh.get(), z_id, &z_ndims)) != NC_NOERR)
                || ((ierr = nc_inq_vardimid(fh.get(), z_id, z_dims)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(fh.get(), z_id, &z_t)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), z_dims[1], &n_z)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_FATAL_ERROR(
                    << "Failed to query mass_vertical_coordinate variable \""
                    << m_z_axis_variable << "\" in file \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return teca_metadata();
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            whole_extent[0] = 0;
            whole_extent[1] = n_x - 1;
            whole_extent[2] = 0;
            whole_extent[3] = n_y - 1;
            whole_extent[4] = 0;
            whole_extent[5] = n_z - 1;

            // TODO -- can we query min max of arrays via netcdf to fill the bounds??

#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // enumerate mesh arrays and their attributes
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
                char var_name[NC_MAX_NAME + 1] = {'\0'};
                int var_type = 0;
                nc_type var_nc_type = 0;
                int n_dims = 0;
                int dim_id[NC_MAX_VAR_DIMS] = {0};
                int n_atts = 0;

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_inq_var(fh.get(), i, var_name,
                        &var_nc_type, &n_dims, dim_id, &n_atts)) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_FATAL_ERROR(
                        << "Failed to query " << i << "th variable, "
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                // convert from the netcdf type code
                NC_DISPATCH(var_nc_type,
                   var_type = teca_variant_array_code<NC_NT>::get();
                   )

                // skip scalars
                if (n_dims == 0)
                    continue;

                std::vector<size_t> dims;
                std::vector<std::string> dim_names;
                for (int ii = 0; ii < n_dims; ++ii)
                {
                    char dim_name[NC_MAX_NAME + 1] = {'\0'};
                    size_t dim = 0;
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if ((ierr = nc_inq_dim(fh.get(), dim_id[ii], dim_name, &dim)) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_FATAL_ERROR(
                            << "Failed to query " << ii << "th dimension of variable, "
                            << var_name << ", " << file << endl << nc_strerror(ierr))
                        return teca_metadata();
                    }
#if !defined(HDF5_THREAD_SAFE)
                    }
#endif
                    dim_names.push_back(dim_name);
                    dims.push_back(dim);
                }

                vars.push_back(var_name);

                teca_metadata atts;
                atts.set("cf_id", i);
                atts.set("cf_dims", dims);
                atts.set("cf_dim_names", dim_names);
                atts.set("cf_type_code", var_nc_type);
                atts.set("type_code", var_type);

                void *att_buffer = nullptr;
                for (int ii = 0; ii < n_atts; ++ii)
                {
                    char att_name[NC_MAX_NAME + 1] = {'\0'};
                    nc_type att_type = 0;
                    size_t att_len = 0;
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if (((ierr = nc_inq_attname(fh.get(), i, ii, att_name)) != NC_NOERR)
                        || ((ierr = nc_inq_att(fh.get(), i, att_name, &att_type, &att_len)) != NC_NOERR))
                    {
                        this->clear_cached_metadata();
                        TECA_FATAL_ERROR("Failed to query " << ii << "th attribute of variable, "
                            << var_name << ", " << file << endl << nc_strerror(ierr))
                        free(att_buffer);
                        return teca_metadata();
                    }
#if !defined(HDF5_THREAD_SAFE)
                    }
#endif
                    if (att_type == NC_CHAR)
                    {
                        char *tmp = static_cast<char*>(realloc(att_buffer, att_len + 1));
                        tmp[att_len] = '\0';
#if !defined(HDF5_THREAD_SAFE)
                        {
                        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                        nc_get_att_text(fh.get(), i, att_name, tmp);
#if !defined(HDF5_THREAD_SAFE)
                        }
#endif
                        teca_netcdf_util::crtrim(tmp, att_len);
                        atts.set(att_name, std::string(tmp));
                        att_buffer = tmp;
                    }
                    else if (att_type == NC_STRING)
                    {
                        char *strs[1] = {nullptr};
#if !defined(HDF5_THREAD_SAFE)
                        {
                        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                        nc_get_att_string(fh.get(), i, att_name, strs);
#if !defined(HDF5_THREAD_SAFE)
                        }
#endif
                        atts.set(att_name, std::string(strs[0]));
#if !defined(HDF5_THREAD_SAFE)
                        {
                        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                        nc_free_string(1, strs);
#if !defined(HDF5_THREAD_SAFE)
                        }
#endif
                    }
                    else
                    {
                        NC_DISPATCH(att_type,
                            NC_NT *tmp = static_cast<NC_NT*>(realloc(att_buffer, sizeof(NC_NT)*att_len));
#if !defined(HDF5_THREAD_SAFE)
                            {
                            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                            nc_get_att(fh.get(), i, att_name, tmp);
#if !defined(HDF5_THREAD_SAFE)
                            }
#endif
                            atts.set(att_name, tmp, att_len);
                            att_buffer = tmp;
                            )
                    }
                }
                free(att_buffer);

                // check if it's a mesh variable, if it is not a mesh variable
                // it is an information variable (ie non-spatial)

                // update the centering
                int centering = teca_array_attributes::no_centering;
                int mesh_dim = 0;
                if (atts.has("stagger") && atts.has("MemoryOrder"))
                {
                    std::string stagger;
                    atts.get("stagger", stagger);

                    std::string mem_order;
                    atts.get("MemoryOrder", mem_order);

                    // check if it's a mesh variable, if it is not a mesh variable
                    // it is an information variable (ie non-spatial)
                    if (mem_order.find("XYZ") != std::string::npos)
                        mesh_dim = 3;
                    else if (mem_order.find("XY") != std::string::npos)
                        mesh_dim = 2;

                    if (mesh_dim)
                    {
                        if (stagger.empty())
                        {
                            centering = teca_array_attributes::cell_centering;
                        }
                        else if (stagger == "X")
                        {
                            centering = teca_array_attributes::x_face_centering;
                        }
                        else if (stagger == "Y")
                        {
                            centering = teca_array_attributes::y_face_centering;
                        }
                        else if (stagger == "Z")
                        {
                            centering = teca_array_attributes::z_face_centering;
                        }
                        else
                        {
                            // handle edge and node if/when they arrise.
                            TECA_FATAL_ERROR(<< stagger << " stagger is not implemented")
                            return teca_metadata();
                        }
                    }
                }
                atts.set("centering", centering);
                atts.set("mesh_dimension", mesh_dim);

                atrs.set(var_name, atts);
            }

            // collect time steps from this and the rest of the files.
            // there are a couple of  performance issues on Lustre.
            // 1) opening a file is slow, there's latency due to contentions
            // 2) reading the time axis is very slow as it's not stored
            //    contiguously by convention. ie. time is an "unlimted"
            //    NetCDF dimension.
            // when procesing large numbers of files these issues kill
            // serial performance. hence we are reading time dimension
            // in parallel.
            using teca_netcdf_util::read_variable;

            read_variable::queue_t thread_pool(MPI_COMM_SELF,
                this->thread_pool_size, true, false);

            // we rely t_axis_variable being empty to indicate either that
            // there is no time axis, or that a time axis will be defined by
            // other algorithm properties. This temporary is used for metadata
            // consistency across those cases.
            std::string t_axis_var = t_axis_variable;

            std::vector<unsigned long> step_count;
            if (!t_axis_variable.empty())
            {
                // assign the reads to threads
                size_t n_files = files.size();
                for (size_t i = 0; i < n_files; ++i)
                {
                    read_variable reader(path, files[i], i, t_axis_variable);
                    read_variable::task_t task(reader);
                    thread_pool.push_task(task);
                }

                // wait for the results
                std::vector<read_variable::data_t> tmp;
                tmp.reserve(n_files);
                thread_pool.wait_all(tmp);

                // unpack the results. map is used to ensure the correct
                // file to time association.
                std::map<unsigned long, p_teca_variant_array>
                    time_arrays(tmp.begin(), tmp.end());
                t_axis = time_arrays[0];
                if (!t_axis)
                {
                    TECA_FATAL_ERROR("Failed to read time axis")
                    return teca_metadata();
                }
                step_count.push_back(time_arrays[0]->size());

                for (size_t i = 1; i < n_files; ++i)
                {
                    p_teca_variant_array tmp = time_arrays[i];
                    t_axis->append(tmp);
                    step_count.push_back(tmp->size());
                }

                // assume that this is a standard calendar if none is provided
                teca_metadata time_atts;
                atrs.get(t_axis_variable, time_atts);
                if (!time_atts.has("calendar"))
                {
                    std::string cal = "standard";
                    if (!this->calendar.empty())
                        cal = "standard";
                    else
                        cal = this->calendar;
                    time_atts.set("calendar", cal);
                    atrs.set(t_axis_variable, time_atts);
                }
            }
            else if (!this->t_values.empty())
            {
                if (this->calendar.empty() || this->t_units.empty())
                {
                    TECA_FATAL_ERROR("calendar and units has to be specified"
                        " for the time variable")
                    return teca_metadata();
                }

                // if time axis is provided manually by the user
                size_t n_t_vals = this->t_values.size();
                if (n_t_vals != files.size())
                {
                    TECA_FATAL_ERROR("Number of files choosen doesn't match the"
                        " number of time values provided")
                    return teca_metadata();
                }

                teca_metadata time_atts;
                time_atts.set("calendar", this->calendar);
                time_atts.set("units", this->t_units);

                atrs.set("time", time_atts);

                p_teca_variant_array_impl<double> t =
                    teca_variant_array_impl<double>::New(n_t_vals, this->t_values.data());

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
                    std::istringstream ss(files[i].c_str());
                    std::tm current_tm;
                    current_tm.tm_year = 0;
                    current_tm.tm_mon = 0;
                    current_tm.tm_mday = 0;
                    current_tm.tm_hour = 0;
                    current_tm.tm_min = 0;
                    current_tm.tm_sec = 0;

                    // attempt to convert the filename into a time
                    ss >> std::get_time(&current_tm,
                        this->filename_time_template.c_str());

                    // check whether the conversion failed
                    if(ss.fail())
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

                // set the time metadata
                teca_metadata time_atts;
                time_atts.set("calendar", calendar);
                time_atts.set("units", t_units);
                atrs.set("time", time_atts);

                // create a teca variant array from the times
                size_t n_t_vals = t_values.size();
                p_teca_variant_array_impl<double> t =
                    teca_variant_array_impl<double>::New(n_t_vals, t_values.data());

                // set the number of time steps
                step_count.resize(n_t_vals, 1);

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
                int x_id = 0;
                nc_type x_t = 0;
                if (((ierr = nc_inq_varid(fh.get(), m_x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                    || ((ierr = nc_inq_vartype(fh.get(), x_id, &x_t)) != NC_NOERR))
                {
                    TECA_FATAL_ERROR("Failed to deduce the time axis type from "
                        << m_x_axis_variable << endl << nc_strerror(ierr))
                    return teca_metadata();
                }

                size_t n_files = files.size();
                NC_DISPATCH_FP(x_t,
                    auto [t, pt] = ::New<NC_TT>(n_files);
                    for (size_t i = 0; i < n_files; ++i)
                    {
                        pt[i] = NC_NT(i);
                        step_count.push_back(1);
                    }
                    t_axis = t;
                    )

                t_axis_var = "time";
            }

            this->internals->metadata.set("variables", vars);
            this->internals->metadata.set("attributes", atrs);

            teca_metadata coords;
            // report which variables will be used to construct the
            // coordinate system
            coords.set("m_x_variable", m_x_axis_variable);
            coords.set("m_y_variable", m_y_axis_variable);
            coords.set("u_x_variable", u_x_axis_variable);
            coords.set("u_y_variable", u_y_axis_variable);
            coords.set("v_x_variable", v_x_axis_variable);
            coords.set("v_y_variable", v_y_axis_variable);
            coords.set("m_z_variable", m_z_axis_variable);
            coords.set("w_z_variable", w_z_axis_variable);
            coords.set("t_variable", t_axis_var);
            coords.set("t", t_axis);
            coords.set("periodic_in_x", this->periodic_in_x);
            coords.set("periodic_in_y", this->periodic_in_y);
            coords.set("periodic_in_z", this->periodic_in_z);
            this->internals->metadata.set("whole_extent", whole_extent);
            //this->internals->metadata.set("bounds", bounds);
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
                "index_request_key", std::string("time_step"));

            this->internals->metadata.to_stream(stream);

#if defined(TECA_HAS_OPENSSL)
            // cache metadata on disk
            bool cached_metadata = false;
            for (int i = n_metadata_cache_paths; i >= 0; --i)
            {
                std::string metadata_cache_file =
                    metadata_cache_path[i] + PATH_SEP + "." + metadata_cache_key + ".tmd";

                if (!teca_file_util::write_stream(metadata_cache_file.c_str(),
                    S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH,
                    "teca_wrf_reader::metadata_cache_file", stream, false))
                {
                    cached_metadata = true;
                    TECA_STATUS("Wrote metadata cache \""
                        << metadata_cache_file << "\"")
                    break;
                }
            }
            if (!cached_metadata)
            {
                TECA_FATAL_ERROR("failed to create a metadata cache")
            }
#endif
        }

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
const_p_teca_dataset teca_wrf_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_wrf_reader::execute" << endl;
#endif
    (void)port;
    (void)input_data;

    // get coordinates
    teca_metadata coords;
    if (this->internals->metadata.get("coordinates", coords))
    {
        TECA_FATAL_ERROR("metadata is missing \"coordinates\"")
        return nullptr;
    }

    p_teca_variant_array in_t;
    if (!(in_t = coords.get("t")))
    {
        TECA_FATAL_ERROR("coordinate metadata is missing time axis")
        return nullptr;
    }

    // get request
    unsigned long time_step = 0;
    double t = 0.0;
    if (!request.get("time", t))
    {
        // translate time to a time step
        VARIANT_ARRAY_DISPATCH_FP(in_t.get(),
            auto [pin_t] = data<TT>(in_t);
            if (teca_coordinate_util::index_of(pin_t, 0,
                in_t->size()-1, static_cast<NT>(t), time_step))
            {
                TECA_FATAL_ERROR("requested time " << t << " not found")
                return nullptr;
            }
            )
    }
    else
    {
        // TODO -- there is currently no error checking here to
        // support case where only 1 time step is present in a file.
        request.get("time_step", time_step);
        if ((in_t) && (time_step < in_t->size()))
            in_t->get(time_step, t);
    }

    unsigned long whole_extent[6] = {0};
    if (this->internals->metadata.get("whole_extent", whole_extent, 6))
    {
        TECA_FATAL_ERROR("time_step=" << time_step
            << " metadata is missing \"whole_extent\"")
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
    }
    else
    {
        // to implement this we'll need to read in all of the coordinate
        // axes, then scan for the bounding indices in the 2d lat, lon, and 3d pd
        // arrays, and then finally apply the subset. this would need to move
        // below since at this point we haven't read the coordinate axes.
        TECA_FATAL_ERROR("bounds_to_extent not implemented for curvilinear meshes")
        return nullptr;
        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        /*if (teca_coordinate_util::bounds_to_extent(
            bounds, in_x, in_y, in_z, extent))
        {
            TECA_FATAL_ERROR("invalid bounds requested.")
            return nullptr;
        }*/
    }

    // get the array attributes
    teca_metadata atrs;
    if (this->internals->metadata.get("attributes", atrs))
    {
        TECA_FATAL_ERROR("time_step=" << time_step
            << " metadata missing \"attributes\"")
        return nullptr;
    }

    // locate file with this time step
    std::vector<unsigned long> step_count;
    if (this->internals->metadata.get("step_count", step_count))
    {
        TECA_FATAL_ERROR("time_step=" << time_step
            << " metadata is missing \"step_count\"")
        return nullptr;
    }

    unsigned long idx = 0;
    unsigned long count = 0;
    for (unsigned int i = 1;
        (i < step_count.size()) && ((count + step_count[i-1]) <= time_step);
        ++idx, ++i)
    {
        count += step_count[i-1];
    }
    unsigned long offs = time_step - count;

    std::string path;
    std::string file;
    if (this->internals->metadata.get("root", path)
        || this->internals->metadata.get("files", idx, file))
    {
        TECA_FATAL_ERROR("time_step=" << time_step
            << " Failed to locate file for time step " << time_step)
        return nullptr;
    }

    // get the file handle for this step
    int ierr = 0;
    std::string file_path = path + PATH_SEP + file;
    teca_netcdf_util::netcdf_handle fh;
    if (fh.open(file_path, NC_NOWRITE))
    {
        TECA_FATAL_ERROR("time_step=" << time_step << " Failed to open \"" << file << "\"")
        return nullptr;
    }
    int file_id = fh.get();

    // requesting arrays is optional, but it's an error
    // to request an array that isn't present.
    std::vector<std::string> arrays;
    request.get("arrays", arrays);
    size_t n_arrays = arrays.size();

    // use  requested arrays to determine which coordinate systems are needed
    int max_mesh_dim = (n_arrays ? 0 : 3);
    int req_mx_coords = (n_arrays ? 0 : 1);
    int req_ux_coords = 0;
    int req_vx_coords = 0;
    int req_mz_coords = (n_arrays ? 0 : 1);
    int req_wz_coords = 0;
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get metadata
        teca_metadata atts;
        int mesh_dim = 0;
        int centering = teca_array_attributes::no_centering;

        if (atrs.get(arrays[i], atts)
            || atts.get("mesh_dimension", mesh_dim)
            || atts.get("centering", centering))
        {
            TECA_FATAL_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
            continue;
        }

        // turn on coordinate systems based on what's requested.
        switch (centering)
        {
            case teca_array_attributes::cell_centering:
                req_mx_coords = 1;
                req_mz_coords = 1;
                break;
            case teca_array_attributes::x_face_centering:
                req_ux_coords = 1;
                req_mz_coords = 1;
                break;
            case teca_array_attributes::y_face_centering:
                req_vx_coords = 1;
                req_mz_coords = 1;
                break;
            case teca_array_attributes::z_face_centering:
                req_mx_coords = 1;
                req_wz_coords = 1;
                break;
            case teca_array_attributes::no_centering:
                continue;
                break;
            default:
                TECA_FATAL_ERROR("Invalid centering " << centering
                    << " array " << i << " \"" << arrays[i])
        }

        // update the max mesh dim.
        max_mesh_dim = std::max(max_mesh_dim, mesh_dim);
    }

    // TODO -- if the variable requested is a 1D or a  2D mesh
    // do we require the requested extent to match or do we
    // fix the extent here?

    // read spatial coordinate arrays
    unsigned long n_m_x = extent[1] - extent[0] + 1;
    unsigned long n_m_y = extent[3] - extent[2] + 1;
    unsigned long n_m_z = extent[5] - extent[4] + 1;
    unsigned long n_m_xy = n_m_x*n_m_y;

    teca_metadata out_atrs;

    // m_x and m_y
    p_teca_variant_array m_x_out;
    p_teca_variant_array m_y_out;
    if (req_mx_coords)
    {
        // collect metadata
        teca_metadata m_x_atts;
        int m_x_type = 0;
        int m_x_id = 0;
        if (atrs.get(m_x_axis_variable, m_x_atts) ||
            m_x_atts.get("cf_type_code", 0, m_x_type) ||
            m_x_atts.get("cf_id", 0, m_x_id))
        {
            TECA_FATAL_ERROR("metadata issue with m_x_axis_variable \""
                << m_x_axis_variable << "\"")
            return nullptr;
        }

        teca_metadata m_y_atts;
        int m_y_type = 0;
        int m_y_id = 0;
        if (atrs.get(m_y_axis_variable, m_y_atts) ||
            m_y_atts.get("cf_type_code", 0, m_y_type) ||
            m_y_atts.get("cf_id", 0, m_y_id))
        {
            TECA_FATAL_ERROR("metadata issue with m_y_axis_variable \""
                << m_y_axis_variable << "\"")
            return nullptr;
        }

        // pass metadata
        out_atrs.set(m_x_axis_variable, m_x_atts);
        out_atrs.set(m_y_axis_variable, m_y_atts);

        // define the hyperslabs we need to read.
        size_t m_x_start[] = {offs, extent[2], extent[0]};
        size_t m_x_count[] = {1, n_m_y, n_m_x};

        NC_DISPATCH_FP(m_x_type,
            // allocate temporary storage
            auto [m_x, pm_x] = ::New<NC_TT>(n_m_xy);
            auto [m_y, pm_y] = ::New<NC_TT>(n_m_xy);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // read the data
            if (((ierr = nc_get_vara(file_id,  m_x_id, m_x_start, m_x_count, pm_x)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read m_x_axis_variable \"" << m_x_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }

            if (((ierr = nc_get_vara(file_id,  m_y_id, m_x_start, m_x_count, pm_y)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read m_y_axis_variable \"" << m_y_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            m_x_out = m_x;
            m_y_out = m_y;
            )
    }

    // u_x and u_y
    p_teca_variant_array u_x_out;
    p_teca_variant_array u_y_out;
    if (req_ux_coords)
    {
        // collect metadata
        teca_metadata u_x_atts;
        int u_x_type = 0;
        int u_x_id = 0;
        if (atrs.get(u_x_axis_variable, u_x_atts) ||
            u_x_atts.get("cf_type_code", 0, u_x_type) ||
            u_x_atts.get("cf_id", 0, u_x_id))
        {
            TECA_FATAL_ERROR("metadata issue with u_x_axis_variable \""
                << u_x_axis_variable << "\"")
            return nullptr;
        }

        teca_metadata u_y_atts;
        int u_y_type = 0;
        int u_y_id = 0;
        if (atrs.get(u_y_axis_variable, u_y_atts) ||
            u_y_atts.get("cf_type_code", 0, u_y_type) ||
            u_y_atts.get("cf_id", 0, u_y_id))
        {
            TECA_FATAL_ERROR("metadata issue with u_y_axis_variable \""
                << u_y_axis_variable << "\"")
            return nullptr;
        }

        // pass metadata
        out_atrs.set(u_x_axis_variable, u_x_atts);
        out_atrs.set(u_y_axis_variable, u_y_atts);

        // define the hyperslabs we need to read.
        size_t n_u_x = n_m_x + 1;
        size_t n_u_xy = n_u_x*n_m_y;
        size_t u_x_start[] = {offs, extent[2], extent[0]};
        size_t u_x_count[] = {1, n_m_y, n_u_x};

        NC_DISPATCH_FP(u_x_type,
            // allocate temporary storage
            auto [u_x, pu_x] = ::New<NC_TT>(n_u_xy);
            auto [u_y, pu_y] = ::New<NC_TT>(n_u_xy);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // read the data
            if (((ierr = nc_get_vara(file_id,  u_x_id, u_x_start, u_x_count, pu_x)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read u_x_axis_variable \"" << u_x_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }

            if (((ierr = nc_get_vara(file_id,  u_y_id, u_x_start, u_x_count, pu_y)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read u_y_axis_variable \"" << u_y_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            u_x_out = u_x;
            u_y_out = u_y;
            )
    }

    // v_x and v_y
    p_teca_variant_array v_x_out;
    p_teca_variant_array v_y_out;
    if (req_vx_coords)
    {
        // collect metadata
        teca_metadata v_x_atts;
        int v_x_type = 0;
        int v_x_id = 0;
        if (atrs.get(v_x_axis_variable, v_x_atts) ||
            v_x_atts.get("cf_type_code", 0, v_x_type) ||
            v_x_atts.get("cf_id", 0, v_x_id))
        {
            TECA_FATAL_ERROR("metadata issue with v_x_axis_variable \""
                << v_x_axis_variable << "\"")
            return nullptr;
        }

        teca_metadata v_y_atts;
        int v_y_type = 0;
        int v_y_id = 0;
        if (atrs.get(v_y_axis_variable, v_y_atts) ||
            v_y_atts.get("cf_type_code", 0, v_y_type) ||
            v_y_atts.get("cf_id", 0, v_y_id))
        {
            TECA_FATAL_ERROR("metadata issue with v_y_axis_variable \""
                << v_y_axis_variable << "\"")
            return nullptr;
        }

        // pass metadata
        out_atrs.set(v_x_axis_variable, v_x_atts);
        out_atrs.set(v_y_axis_variable, v_y_atts);

        // define the hyperslabs we need to read.
        size_t n_v_y = n_m_y + 1;
        size_t n_v_xy = n_m_x*n_v_y;
        size_t v_x_start[] = {offs, extent[2], extent[0]};
        size_t v_x_count[] = {1, n_v_y, n_m_x};

        NC_DISPATCH_FP(v_x_type,
            // allocate temporary storage
            auto [v_x, pv_x] = ::New<NC_TT>(n_v_xy);
            auto [v_y, pv_y] = ::New<NC_TT>(n_v_xy);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // read the data
            if (((ierr = nc_get_vara(file_id,  v_x_id, v_x_start, v_x_count, pv_x)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read v_x_axis_variable \"" << v_x_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }

            if (((ierr = nc_get_vara(file_id,  v_y_id, v_x_start, v_x_count, pv_y)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read v_y_axis_variable \"" << v_y_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            v_x_out = v_x;
            v_y_out = v_y;
            )
    }

    // m_z
    p_teca_variant_array m_z_out;
    if (req_mz_coords)
    {
        // collect metadata
        teca_metadata m_z_atts;
        int m_z_type = 0;
        int m_z_id = 0;
        if (atrs.get(m_z_axis_variable, m_z_atts) ||
            m_z_atts.get("cf_type_code", 0, m_z_type) ||
            m_z_atts.get("cf_id", 0, m_z_id))
        {
            TECA_FATAL_ERROR("mm_zdata issue with m_z_axis_variable \""
                << m_z_axis_variable << "\"")
            return nullptr;
        }

        // pass metadata
        out_atrs.set(m_z_axis_variable, m_z_atts);

        // define the hyperslabs we need to read.
        size_t m_z_start[] = {offs, extent[4]};
        size_t m_z_count[] = {1, n_m_z};

        NC_DISPATCH_FP(m_z_type,
            // allocate temporary storage
            auto [m_z, pm_z] = ::New<NC_TT>(n_m_z);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // read the data
            if (((ierr = nc_get_vara(file_id,  m_z_id, m_z_start, m_z_count, pm_z)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read m_z_axis_variable \""
                    << m_z_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            m_z_out = m_z;
            )
    }

    // w_z
    p_teca_variant_array w_z_out;
    if (req_wz_coords)
    {
        // collect metadata
        teca_metadata w_z_atts;
        int w_z_type = 0;
        int w_z_id = 0;
        if (atrs.get(w_z_axis_variable, w_z_atts) ||
            w_z_atts.get("cf_type_code", 0, w_z_type) ||
            w_z_atts.get("cf_id", 0, w_z_id))
        {
            TECA_FATAL_ERROR("mw_zdata issue with w_z_axis_variable \""
                << w_z_axis_variable << "\"")
            return nullptr;
        }

        // pass metadata
        out_atrs.set(w_z_axis_variable, w_z_atts);

        // define the hyperslabs we need to read.
        size_t n_w_z = n_m_z + 1;
        size_t w_z_start[] = {offs, extent[4]};
        size_t w_z_count[] = {1, n_w_z};

        NC_DISPATCH_FP(w_z_type,
            // allocate temporary storage
            auto [w_z, pw_z] = ::New<NC_TT>(n_w_z);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            // read the data
            if (((ierr = nc_get_vara(file_id,  w_z_id, w_z_start, w_z_count, pw_z)) != NC_NOERR))
            {
                TECA_FATAL_ERROR("At time_step " << time_step
                    << " failed to read w_z_axis_variable \""
                    << w_z_axis_variable << "\" from \""
                    << file << "\"" << endl << nc_strerror(ierr))
                return nullptr;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            w_z_out = w_z;
            )
    }

    // create output dataset
    p_teca_arakawa_c_grid mesh = teca_arakawa_c_grid::New();

    mesh->set_m_x_coordinates(m_x_axis_variable, m_x_out);
    mesh->set_m_y_coordinates(m_y_axis_variable, m_y_out);
    mesh->set_u_x_coordinates(u_x_axis_variable, u_x_out);
    mesh->set_u_y_coordinates(u_y_axis_variable, u_y_out);
    mesh->set_v_x_coordinates(v_x_axis_variable, v_x_out);
    mesh->set_v_y_coordinates(v_y_axis_variable, v_y_out);
    mesh->set_m_z_coordinates(m_z_axis_variable, m_z_out);
    mesh->set_w_z_coordinates(w_z_axis_variable, w_z_out);

    p_teca_double_array t_out = teca_double_array::New(1, t);
    mesh->set_t_coordinates(t_axis_variable, t_out);
    mesh->set_request_index("time_step", time_step);
    mesh->set_time(t);
    mesh->set_time_step(time_step);
    mesh->set_whole_extent(whole_extent);
    mesh->set_extent(extent);
    mesh->set_bounds(bounds);
    mesh->set_periodic_in_x(this->periodic_in_x);
    mesh->set_periodic_in_y(this->periodic_in_y);
    mesh->set_periodic_in_z(this->periodic_in_z);

    // pass time axis attributes
    teca_metadata time_atts;
    std::string calendar;
    std::string units;
    if (!atrs.get(t_axis_variable, time_atts)
       && !time_atts.get("calendar", calendar)
       && !time_atts.get("units", units))
    {
        mesh->set_calendar(calendar);
        mesh->set_time_units(units);
    }

    // pass the attributes for the arrays read
    for (unsigned int i = 0; i < n_arrays; ++i)
        out_atrs.set(arrays[i], atrs.get(arrays[i]));

    mesh->set_attributes(out_atrs);

    // read requested arrays
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get metadata
        teca_metadata atts;
        int type = 0;
        int id = 0;
        int mesh_dim = 0;
        int centering = teca_array_attributes::no_centering;
        p_teca_size_t_array dims;

        if (atrs.get(arrays[i], atts)
            || atts.get("cf_type_code", 0, type)
            || atts.get("cf_id", 0, id)
            || atts.get("mesh_dimension", mesh_dim)
            || atts.get("centering", centering)
            || teca_coordinate_util::validate_centering(centering)
            || !(dims = std::dynamic_pointer_cast<teca_size_t_array>(atts.get("cf_dims"))))
        {
            TECA_FATAL_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
            continue;
        }

        // read requested variables
        if (mesh_dim > 0)
        {
            size_t var_extent[6];
            memcpy(var_extent, extent, sizeof(size_t)*6);

            if ((centering != teca_array_attributes::point_centering) &&
                teca_coordinate_util::convert_cell_extent(var_extent, centering))
            {
                TECA_FATAL_ERROR("Failed to convert the extent for \""
                    << arrays[i] << "\"")
                continue;
            }

            // figure out the mapping between our extent and netcdf
            // representation
            std::vector<size_t> starts;
            std::vector<size_t> counts;
            size_t count = 0;
            size_t mesh_size = 1;
            if (!t_axis_variable.empty())
            {
                starts.push_back(offs);
                counts.push_back(1);
            }

            if ((mesh_dim == 1) || (mesh_dim == 3))
            {
                starts.push_back(var_extent[4]);
                count = var_extent[5] - var_extent[4] + 1;
                counts.push_back(count);
                mesh_size *= count;
            }

            if ((mesh_dim == 2) || (mesh_dim == 3))
            {
                starts.push_back(var_extent[2]);
                count = var_extent[3] - var_extent[2] + 1;
                counts.push_back(count);
                mesh_size *= count;

                starts.push_back(var_extent[0]);
                count = var_extent[1] - var_extent[0] + 1;
                counts.push_back(count);
                mesh_size *= count;
            }

            // read mesh based data
            p_teca_variant_array array;
            NC_DISPATCH(type,
                auto [a, pa] = ::New<NC_TT>(mesh_size);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], pa)) != NC_NOERR)
                {
                    TECA_FATAL_ERROR("time_step=" << time_step
                        << " Failed to read variable \"" << arrays[i] << "\" "
                        << file << endl << nc_strerror(ierr))
                    continue;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                array = a;
                )

            mesh->get_arrays(centering)->append(arrays[i], array);
        }
        else
        {
            // read non-spatial data
            // if the first dimension is time then select the requested time
            // step. otherwise read the entire thing
            size_t n_dims = dims->size();
            std::vector<size_t> starts(n_dims);
            std::vector<size_t> counts(n_dims);
            std::string dim_0_name;
            if (atts.get("cf_dim_names", 0, dim_0_name))
            {
                TECA_FATAL_ERROR("Failed to get dim 0 name")
                return nullptr;
            }
            size_t n_vals = 1;
            if (!t_axis_variable.empty() && (dim_0_name == "Time"))
            {
                starts[0] = offs;
                counts[0] = 1;
            }
            else
            {
                starts[0] = 0;
                size_t dim_len = dims->get(0);
                counts[0] = dim_len;
                n_vals = dim_len;
            }

            for (unsigned int ii = 1; ii < n_dims; ++ii)
            {
                size_t dim_len = dims->get(ii);
                counts[ii] = dim_len;
                n_vals *= dim_len;
            }

            p_teca_variant_array array;

            NC_DISPATCH(type,
                auto [a, pa] = ::New<NC_TT>(n_vals);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], pa)) != NC_NOERR)
                {
                    TECA_FATAL_ERROR("time_step=" << time_step
                        << " Failed to read \"" << arrays[i] << "\" "
                        << file << endl << nc_strerror(ierr))
                    continue;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                array = a;
                )

            mesh->get_information_arrays()->append(arrays[i], array);
        }
    }

    return mesh;
}
