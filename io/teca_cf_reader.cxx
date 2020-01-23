#include "teca_cf_reader.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh.h"
#include "teca_thread_pool.h"
#include "teca_coordinate_util.h"
#include "teca_netcdf_util.h"
#include "calcalcs.h"

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

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_OPENSSL)
#include <openssl/sha.h>
#endif

// data and task types
using read_variable_data_t = std::pair<unsigned long, p_teca_variant_array>;
using read_variable_task_t = std::packaged_task<read_variable_data_t()>;

using read_variable_queue_t =
    teca_thread_pool<read_variable_task_t, read_variable_data_t>;

using p_read_variable_queue_t = std::shared_ptr<read_variable_queue_t>;

// internals for the cf reader
class teca_cf_reader_internals
{
public:
    teca_cf_reader_internals()
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
std::string teca_cf_reader_internals::create_metadata_cache_key(
    const std::string &path, const std::vector<std::string> &files)
{
    // create the hash using the file names and path
    SHA_CTX ctx;
    SHA1_Init(&ctx);

    SHA1_Update(&ctx, path.c_str(), path.size());

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

// function that reads and returns a variable from the
// named file. we're doing this so we can do thread
// parallel I/O to hide some of the cost of opening files
// on Lustre and to hide the cost of reading time coordinate
// which is typically very expensive as NetCDF stores
// unlimted dimensions non-contiguously
class read_variable
{
public:
    read_variable(const std::string &path, const std::string &file,
        unsigned long id, const std::string &variable) : m_path(path),
        m_file(file), m_variable(variable), m_id(id)
    {}

    std::pair<unsigned long, p_teca_variant_array> operator()()
    {
        p_teca_variant_array var;

        // get a handle to the file. managed by the reader
        // since it will reuse the handle when it needs to read
        // mesh based data
        std::string file_path = m_path + PATH_SEP + m_file;

        teca_netcdf_util::netcdf_handle fh;
        if (fh.open(file_path, NC_NOWRITE))
        {
            TECA_ERROR("Failed to open read variable \"" << m_variable
                << "\" from \"" << m_file << "\"")
            return std::make_pair(m_id, nullptr);
        }

        // query variable attributes
        int file_id = fh.get();
        int var_id = 0;
        size_t var_size = 0;
        nc_type var_type = 0;

        int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if (((ierr = nc_inq_dimid(file_id, m_variable.c_str(), &var_id)) != NC_NOERR)
            || ((ierr = nc_inq_dimlen(file_id, var_id, &var_size)) != NC_NOERR)
            || ((ierr = nc_inq_varid(file_id, m_variable.c_str(), &var_id)) != NC_NOERR)
            || ((ierr = nc_inq_vartype(file_id, var_id, &var_type)) != NC_NOERR))
        {
            TECA_ERROR("Failed to read metadata for variable \"" << m_variable
                << "\" from \"" << m_file << "\". " << nc_strerror(ierr))
            return std::make_pair(m_id, nullptr);
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif

        // allocate a buffer and read the variable.
        NC_DISPATCH(var_type,
            size_t start = 0;
            p_teca_variant_array_impl<NC_T> var = teca_variant_array_impl<NC_T>::New();
            var->resize(var_size);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_get_vara(file_id, var_id, &start, &var_size, var->get())) != NC_NOERR)
            {
                TECA_ERROR("Failed to read variable \"" << m_variable  << "\" from \""
                    << m_file << "\". " << nc_strerror(ierr))
                return std::make_pair(m_id, nullptr);
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            // success!
            return std::make_pair(m_id, var);
            )

        // unsupported type
        TECA_ERROR("Failed to read variable \"" << m_variable
            << "\" from \"" << m_file << "\". Unsupported data type")
        return std::make_pair(m_id, nullptr);
    }

private:
    std::string m_path;
    std::string m_file;
    std::string m_variable;
    unsigned long m_id;
};





// --------------------------------------------------------------------------
teca_cf_reader::teca_cf_reader() :
    files_regex(""),
    x_axis_variable("lon"),
    y_axis_variable("lat"),
    z_axis_variable(""),
    t_axis_variable("time"),
    t_calendar(""),
    t_units(""),
    filename_time_template(""),
    periodic_in_x(0),
    periodic_in_y(0),
    periodic_in_z(0),
    thread_pool_size(-1),
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
        TECA_POPTS_GET(std::vector<std::string>, prefix, file_names,
            "paths/file names to read")
        TECA_POPTS_GET(std::string, prefix, files_regex,
            "a regular expression that matches the set of files "
            "comprising the dataset")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "name of variable that has x axis coordinates (lon)")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "name of variable that has y axis coordinates (lat)")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "name of variable that has z axis coordinates ()")
        TECA_POPTS_GET(std::string, prefix, t_axis_variable,
            "name of variable that has t axis coordinates (time)")
        TECA_POPTS_GET(std::string, prefix, t_calendar,
            "name of variable that has the time calendar (calendar)")
        TECA_POPTS_GET(std::string, prefix, t_units,
            "a std::get_time template for decoding time from the input filename")
        TECA_POPTS_GET(std::string, prefix, filename_time_template,
            "name of variable that has the time unit (units)")
        TECA_POPTS_GET(std::vector<double>, prefix, t_values,
            "name of variable that has t axis values set by the"
            "the user if the file doesn't have time variable set ()")
        TECA_POPTS_GET(int, prefix, periodic_in_x,
            "the dataset has apriodic boundary in the x direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_y,
            "the dataset has apriodic boundary in the y direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_z,
            "the dataset has apriodic boundary in the z direction (0)")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "set the number of I/O threads (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, file_names)
    TECA_POPTS_SET(opts, std::string, prefix, files_regex)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_calendar)
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
                TECA_ERROR(
                    << "Failed to locate any files" << endl
                    << this->files_regex << endl
                    << path << endl
                    << regex)
                return teca_metadata();
            }
        }

#if defined(TECA_HAS_OPENSSL)
        // look for a metadata cache. we are caching it on disk
        // as for large datasets on Lustre, scanning the time
        // dimension is costly because of NetCDF CF convention
        // that time is unlimitted and thus not layed out contiguously
        // in the files.
        std::string metadata_cache_key =
            this->internals->create_metadata_cache_key(path, files);

        std::string metadata_cache_path[3] =
            {path, ".", (getenv("HOME") ? : ".")};

        for (int i = 0; i < 3; ++i)
        {
            std::string metadata_cache_file =
                metadata_cache_path[i] + PATH_SEP + metadata_cache_key + ".md";

            if (teca_file_util::file_exists(metadata_cache_file.c_str()))
            {
                // read the cache
                if (teca_file_util::read_stream(metadata_cache_file.c_str(),
                    "teca_cf_reader::metadata_cache_file", stream))
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
            int y_id = 0;
            int z_id = 0;
            size_t n_x = 1;
            size_t n_y = 1;
            size_t n_z = 1;
            nc_type x_t = 0;
            nc_type y_t = 0;
            nc_type z_t = 0;
            int n_vars = 0;
            p_teca_variant_array x_axis;
            p_teca_variant_array y_axis;
            p_teca_variant_array z_axis;
            p_teca_variant_array t_axis;
            double bounds[6] = {0.0};
            unsigned long whole_extent[6] = {0ul};

            teca_metadata atrs;
            std::vector<std::string> vars;

            teca_netcdf_util::netcdf_handle fh;
            if (fh.open(file.c_str(), NC_NOWRITE))
            {
                TECA_ERROR("Failed to open " << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }

            // query mesh axes
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if (((ierr = nc_inq_dimid(fh.get(), x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), x_id, &n_x)) != NC_NOERR)
                || ((ierr = nc_inq_varid(fh.get(), x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(fh.get(), x_id, &x_t)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query x axis variable \"" << x_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }
#if !defined(HDF5_THREAD_SAFE)
            }
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif

            if (!y_axis_variable.empty()
                && (((ierr = nc_inq_dimid(fh.get(), y_axis_variable.c_str(), &y_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), y_id, &n_y)) != NC_NOERR)
                || ((ierr = nc_inq_varid(fh.get(), y_axis_variable.c_str(), &y_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(fh.get(), y_id, &y_t)) != NC_NOERR)))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query y axis variable \"" << y_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }
#if !defined(HDF5_THREAD_SAFE)
            }
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif

            if (!z_axis_variable.empty()
                && (((ierr = nc_inq_dimid(fh.get(), z_axis_variable.c_str(), &z_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(fh.get(), z_id, &n_z)) != NC_NOERR)
                || ((ierr = nc_inq_varid(fh.get(), z_axis_variable.c_str(), &z_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(fh.get(), z_id, &z_t)) != NC_NOERR)))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query z axis variable \"" << z_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }
#if !defined(HDF5_THREAD_SAFE)
            }
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif

            // enumerate mesh arrays and their attributes
            if (((ierr = nc_inq_nvars(fh.get(), &n_vars)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
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
                nc_type var_type = 0;
                int n_dims = 0;
                int dim_id[NC_MAX_VAR_DIMS] = {0};
                int n_atts = 0;

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_inq_var(fh.get(), i, var_name,
                        &var_type, &n_dims, dim_id, &n_atts)) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_ERROR(
                        << "Failed to query " << i << "th variable, "
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif

                // skip scalars
                if (n_dims == 0)
                    continue;

                std::vector<size_t> dims;
                std::vector<std::string> dim_names;
                std::string centering("point");
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
                        TECA_ERROR(
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
                atts.set("id", i);
                atts.set("dims", dims);
                atts.set("dim_names", dim_names);
                atts.set("type", var_type);
                atts.set("centering", centering);

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
                        TECA_ERROR("Failed to query " << ii << "th attribute of variable, "
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
                            NC_T *tmp = static_cast<NC_T*>(realloc(att_buffer, att_len));
#if !defined(HDF5_THREAD_SAFE)
                            {
                            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                            nc_get_att(fh.get(), i, att_name, tmp);
#if !defined(HDF5_THREAD_SAFE)
                            }
#endif
                            atts.set(att_name, tmp, att_len);
                            )
                    }
                }
                free(att_buffer);

                atrs.set(var_name, atts);
            }

            // read spatial coordinate arrays
            NC_DISPATCH_FP(x_t,
                size_t x_0 = 0;
                p_teca_variant_array_impl<NC_T> x = teca_variant_array_impl<NC_T>::New(n_x);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(fh.get(), x_id, &x_0, &n_x, x->get())) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_ERROR(
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

            if (!y_axis_variable.empty())
            {
                NC_DISPATCH_FP(y_t,
                    size_t y_0 = 0;
                    p_teca_variant_array_impl<NC_T> y = teca_variant_array_impl<NC_T>::New(n_y);
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if ((ierr = nc_get_vara(fh.get(), y_id, &y_0, &n_y, y->get())) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
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
            }
            else
            {
                NC_DISPATCH_FP(x_t,
                    p_teca_variant_array_impl<NC_T> y = teca_variant_array_impl<NC_T>::New(1);
                    y->set(0, NC_T());
                    y_axis = y;
                    )
            }

            if (!z_axis_variable.empty())
            {
                NC_DISPATCH_FP(z_t,
                    size_t z_0 = 0;
                    p_teca_variant_array_impl<NC_T> z = teca_variant_array_impl<NC_T>::New(n_z);
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if ((ierr = nc_get_vara(fh.get(), z_id, &z_0, &n_z, z->get())) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
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
            }
            else
            {
                NC_DISPATCH_FP(x_t,
                    p_teca_variant_array_impl<NC_T> z = teca_variant_array_impl<NC_T>::New(1);
                    z->set(0, NC_T());
                    z_axis = z;
                    )
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
            read_variable_queue_t thread_pool(MPI_COMM_SELF,
                this->thread_pool_size, true, false);

            std::vector<unsigned long> step_count;
            if (!t_axis_variable.empty())
            {
                // assign the reads to threads
                size_t n_files = files.size();
                for (size_t i = 0; i < n_files; ++i)
                {
                    read_variable reader(path, files[i], i, this->t_axis_variable);
                    read_variable_task_t task(reader);
                    thread_pool.push_task(task);
                }

                // wait for the results
                std::vector<read_variable_data_t> tmp;
                tmp.reserve(n_files);
                thread_pool.wait_data(tmp);

                // unpack the results. map is used to ensure the correct
                // file to time association.
                std::map<unsigned long, p_teca_variant_array>
                    time_arrays(tmp.begin(), tmp.end());
                t_axis = time_arrays[0];
                if (!t_axis)
                {
                    TECA_ERROR("Failed to read time axis")
                    return teca_metadata();
                }
                step_count.push_back(time_arrays[0]->size());

                for (size_t i = 1; i < n_files; ++i)
                {
                    p_teca_variant_array tmp = time_arrays[i];
                    t_axis->append(*tmp.get());
                    step_count.push_back(tmp->size());
                }
            }
            else if (!this->t_values.empty())
            {
                if (this->t_calendar.empty() || this->t_units.empty())
                {
                    TECA_ERROR("calendar and units has to be specified"
                        " for the time variable")
                    return teca_metadata();
                }

                // if time axis is provided manually by the user
                size_t n_t_vals = this->t_values.size();
                if (n_t_vals != files.size())
                {
                    TECA_ERROR("Number of files choosen doesn't match the"
                        " number of time values provided")
                    return teca_metadata();
                }

                teca_metadata time_atts;
                time_atts.set("calendar", this->t_calendar);
                time_atts.set("units", this->t_units);

                atrs.set("time", time_atts);

                p_teca_variant_array_impl<double> t =
                    teca_variant_array_impl<double>::New(this->t_values.data(), n_t_vals);

                step_count.resize(n_t_vals, 1);

                t_axis = t;
            }
            // infer the time from the filenames
            else if (! this->filename_time_template.empty())
            {
                std::vector<double> t_values;

                std::string t_units = this->t_units;
                std::string t_calendar = this->t_calendar;

                // assume that this is a standard calendar if none is provided
                if (this->t_calendar.empty())
                {
                    t_calendar = "standard";
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
                        TECA_ERROR("Failed to infer time from filename \"" <<
                            files[i] << "\" using format \"" <<
                            this->filename_time_template << "\"")
                        return teca_metadata();
                    }

                    // set the time units based on the first file date if we
                    // don't have time units
                    if (t_units.empty() and i == 0)
                    {
                        std::string t_units_fmt =
                            "days since %Y-%m-%d 00:00:00";

                        // convert the time data to a string
                        char tmp[256];
                        if (strftime(tmp, sizeof(tmp), t_units_fmt.c_str(),
                              &current_tm) == 0)
                        {
                            TECA_ERROR(
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
                    if (calcalcs::coordinate(year, mon, day, hour, minute,
                          second, t_units.c_str(), t_calendar.c_str(),
                          &current_time))
                    {
                        TECA_ERROR(
                            "conversion of date inferred from filename failed");
                    }
#else
                    (void)date;
                    TECA_ERROR(
                        "The UDUnits package is required for this operation")
                    return -1;
#endif
                    // add the current time to the list
                    t_values.push_back(current_time);
                }

                // set the time metadata
                teca_metadata time_atts;
                time_atts.set("calendar", t_calendar);
                time_atts.set("units", t_units);
                atrs.set("time", time_atts);

                // create a teca variant array from the times
                size_t n_t_vals = t_values.size();
                p_teca_variant_array_impl<double> t =
                    teca_variant_array_impl<double>::New(t_values.data(),
                            n_t_vals);

                // set the number of time steps
                step_count.resize(n_t_vals, 1);

                // set the time axis
                t_axis = t;
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
                NC_DISPATCH_FP(x_t,
                    p_teca_variant_array_impl<NC_T> t =
                        teca_variant_array_impl<NC_T>::New(n_files);
                    for (size_t i = 0; i < n_files; ++i)
                    {
                        t->set(i, NC_T(i));
                        step_count.push_back(1);
                    }
                    t_axis = t;
                    )
            }

            this->internals->metadata.set("variables", vars);
            this->internals->metadata.set("attributes", atrs);

            teca_metadata coords;
            coords.set("x_variable", x_axis_variable);
            coords.set("y_variable",
                    (y_axis_variable.empty() ? "y" : y_axis_variable));
            coords.set("z_variable",
                    (z_axis_variable.empty() ? "z" : z_axis_variable));
            coords.set("t_variable",
                    (t_axis_variable.empty() ? "t" : t_axis_variable));
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
                "index_request_key", std::string("time_step"));

            this->internals->metadata.to_stream(stream);

#if defined(TECA_HAS_OPENSSL)
            // cache metadata on disk
            bool cached_metadata = false;
            for (int i = 0; i < 3; ++i)
            {
                std::string metadata_cache_file =
                    metadata_cache_path[i] + PATH_SEP + metadata_cache_key + ".md";

                if (!teca_file_util::write_stream(metadata_cache_file.c_str(),
                    "teca_cf_reader::metadata_cache_file", stream, false))
                {
                    cached_metadata = true;
                    TECA_STATUS("Wrote metadata cache \""
                        << metadata_cache_file << "\"")
                    break;
                }
            }
            if (!cached_metadata)
            {
                TECA_ERROR("failed to create a metadata cache")
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

    // get coordinates
    teca_metadata coords;
    if (this->internals->metadata.get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing \"coordinates\"")
        return nullptr;
    }

    p_teca_variant_array in_x, in_y, in_z, in_t;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")) || !(in_t = coords.get("t")))
    {
        TECA_ERROR("metadata is missing coordinate arrays")
        return nullptr;
    }

    // get request
    unsigned long time_step = 0;
    double t = 0.0;
    if (!request.get("time", t))
    {
        // translate time to a time step
        TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
            in_t.get(),

            NT *pin_t = dynamic_cast<TT*>(in_t.get())->get();

            if (teca_coordinate_util::index_of(pin_t, 0,
                in_t->size()-1, static_cast<NT>(t), time_step))
            {
                TECA_ERROR("requested time " << t << " not found")
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
        TECA_ERROR("time_step=" << time_step
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
        // get bounds of the extent being read
        for (int i = 0; i < 6; ++i)
            in_x->get(extent[i], bounds[i]);
    }
    else
    {
        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        if (teca_coordinate_util::bounds_to_extent(
            bounds, in_x, in_y, in_z, extent))
        {
            TECA_ERROR("invalid bounds requested.")
            return nullptr;
        }
    }

    // requesting arrays is optional, but it's an error
    // to request an array that isn't present.
    std::vector<std::string> arrays;
    request.get("arrays", arrays);
    size_t n_arrays = arrays.size();

    // slice axes on the requested extent
    p_teca_variant_array out_x = in_x->new_copy(extent[0], extent[1]);
    p_teca_variant_array out_y = in_y->new_copy(extent[2], extent[3]);
    p_teca_variant_array out_z = in_z->new_copy(extent[4], extent[5]);

    // locate file with this time step
    std::vector<unsigned long> step_count;
    if (this->internals->metadata.get("step_count", step_count))
    {
        TECA_ERROR("time_step=" << time_step
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
        TECA_ERROR("time_step=" << time_step
            << " Failed to locate file for time step " << time_step)
        return nullptr;
    }

    // get the file handle for this step
    int ierr = 0;
    std::string file_path = path + PATH_SEP + file;
    teca_netcdf_util::netcdf_handle fh;
    if (fh.open(file_path, NC_NOWRITE))
    {
        TECA_ERROR("time_step=" << time_step << " Failed to open \"" << file << "\"")
        return nullptr;
    }
    int file_id = fh.get();

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates(x_axis_variable, out_x);
    mesh->set_y_coordinates(y_axis_variable, out_y);
    mesh->set_z_coordinates(z_axis_variable, out_z);
    mesh->set_time(t);
    mesh->set_time_step(time_step);
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
        TECA_ERROR("time_step=" << time_step
            << " metadata missing \"attributes\"")
        return nullptr;
    }

    // pass time axis attributes
    teca_metadata time_atts;
    std::string calendar;
    std::string units;
    if (!atrs.get("time", time_atts)
       && !time_atts.get("calendar", calendar)
       && !time_atts.get("units", units))
    {
        mesh->set_calendar(calendar);
        mesh->set_time_units(units);
    }

    // add the pipeline keys
    teca_metadata &md = mesh->get_metadata();
    md.set("index_request_key", std::string("time_step"));
    md.set("time_step", time_step);

    // pass the attributes for the arrays read
    teca_metadata out_atrs;
    for (unsigned int i = 0; i < n_arrays; ++i)
        out_atrs.set(arrays[i], atrs.get(arrays[i]));

    // pass coordinate axes attributes
    if (atrs.has(x_axis_variable))
        out_atrs.set(x_axis_variable, atrs.get(x_axis_variable));
    if (atrs.has(y_axis_variable))
        out_atrs.set(y_axis_variable, atrs.get(y_axis_variable));
    if (atrs.has(z_axis_variable))
        out_atrs.set(z_axis_variable, atrs.get(z_axis_variable));
    if (!time_atts.empty())
        out_atrs.set("time", time_atts);

    mesh->get_metadata().set("attributes", out_atrs);

    // figure out the mapping between our extent and netcdf
    // representation
    std::vector<std::string> mesh_dim_names;
    std::vector<size_t> starts;
    std::vector<size_t> counts;
    size_t mesh_size = 1;
    if (!t_axis_variable.empty())
    {
        mesh_dim_names.push_back(t_axis_variable);
        starts.push_back(offs);
        counts.push_back(1);
    }
    if (!z_axis_variable.empty())
    {
        mesh_dim_names.push_back(z_axis_variable);
        starts.push_back(extent[4]);
        size_t count = extent[5] - extent[4] + 1;
        counts.push_back(count);
        mesh_size *= count;
    }
    if (!y_axis_variable.empty())
    {
        mesh_dim_names.push_back(y_axis_variable);
        starts.push_back(extent[2]);
        size_t count = extent[3] - extent[2] + 1;
        counts.push_back(count);
        mesh_size *= count;
    }
    if (!x_axis_variable.empty())
    {
        mesh_dim_names.push_back(x_axis_variable);
        starts.push_back(extent[0]);
        size_t count = extent[1] - extent[0] + 1;
        counts.push_back(count);
        mesh_size *= count;
    }

    // read requested arrays
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get metadata
        teca_metadata atts;
        int type = 0;
        int id = 0;
        p_teca_size_t_array dims;
        p_teca_string_array dim_names;

        if (atrs.get(arrays[i], atts)
            || atts.get("type", 0, type)
            || atts.get("id", 0, id)
            || !(dims = std::dynamic_pointer_cast<teca_size_t_array>(atts.get("dims")))
            || !(dim_names = std::dynamic_pointer_cast<teca_string_array>(atts.get("dim_names"))))
        {
            TECA_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
            continue;
        }

        // check if it's a mesh variable, if it is not a mesh variable
        // it is an information variable (ie non-spatiol)
        bool mesh_var = false;
        unsigned int n_dims = dim_names->size();

        if (n_dims == mesh_dim_names.size())
        {
            mesh_var = true;
            for (unsigned int ii = 0; ii < n_dims; ++ii)
            {
                if (dim_names->get(ii) != mesh_dim_names[ii])
                {
                    mesh_var = false;
                    break;
                }
            }
        }

        // read requested variables
        if (mesh_var)
        {
            // read mesh based data
            p_teca_variant_array array;
            NC_DISPATCH(type,
                p_teca_variant_array_impl<NC_T> a = teca_variant_array_impl<NC_T>::New(mesh_size);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], a->get())) != NC_NOERR)
                {
                    TECA_ERROR("time_step=" << time_step
                        << " Failed to read variable \"" << arrays[i] << "\" "
                        << file << endl << nc_strerror(ierr))
                    continue;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                array = a;
                )
            mesh->get_point_arrays()->append(arrays[i], array);
        }
        else
        {
            // read non-spatial data
            // if the first dimension is time then select the requested time
            // step. otherwise read the entire thing
            std::vector<size_t> starts(n_dims);
            std::vector<size_t> counts(n_dims);
            size_t n_vals = 1;
            if (dim_names->get(0) == this->t_axis_variable)
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
                p_teca_variant_array_impl<NC_T> a = teca_variant_array_impl<NC_T>::New(n_vals);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], a->get())) != NC_NOERR)
                {
                    TECA_ERROR("time_step=" << time_step
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
