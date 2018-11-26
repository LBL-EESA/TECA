#include "teca_cf_reader.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh.h"
#include "teca_thread_pool.h"
#include "teca_coordinate_util.h"

#include <netcdf.h>
#include <iostream>
#include <algorithm>
#include <cstring>
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

// macro to help with netcdf data types
#define NC_DISPATCH_FP(tc_, code_)                          \
    switch (tc_)                                            \
    {                                                       \
    NC_DISPATCH_CASE(NC_FLOAT, float, code_)                \
    NC_DISPATCH_CASE(NC_DOUBLE, double, code_)              \
    default:                                                \
        TECA_ERROR("netcdf type code_ " << tc_              \
            << " is not a floating point type")             \
    }
#define NC_DISPATCH(tc_, code_)                             \
    switch (tc_)                                            \
    {                                                       \
    NC_DISPATCH_CASE(NC_BYTE, char, code_)                  \
    NC_DISPATCH_CASE(NC_UBYTE, unsigned char, code_)        \
    NC_DISPATCH_CASE(NC_CHAR, char, code_)                  \
    NC_DISPATCH_CASE(NC_SHORT, short int, code_)            \
    NC_DISPATCH_CASE(NC_USHORT, unsigned short int, code_)  \
    NC_DISPATCH_CASE(NC_INT, int, code_)                    \
    NC_DISPATCH_CASE(NC_UINT, unsigned int, code_)          \
    NC_DISPATCH_CASE(NC_INT64, long long, code_)            \
    NC_DISPATCH_CASE(NC_UINT64, unsigned long long, code_)  \
    NC_DISPATCH_CASE(NC_FLOAT, float, code_)                \
    NC_DISPATCH_CASE(NC_DOUBLE, double, code_)              \
    default:                                                \
        TECA_ERROR("netcdf type code_ " << tc_              \
            << " is not supported")                         \
    }
#define NC_DISPATCH_CASE(cc_, tt_, code_)   \
    case cc_:                               \
    {                                       \
        using NC_T = tt_;                   \
        code_                               \
        break;                              \
    }

#if !defined(HDF5_THREAD_SAFE)
// NetCDF 3 is not threadsafe. The HDF5 C-API can be compiled to be threadsafe,
// but it is usually not. NetCDF uses HDF5-HL API to access HDF5, but HDF5-HL
// API is not threadsafe without the --enable-unsupported flag. For all those
// reasons it's best for the time being to protect all NetCDF I/O.
static std::mutex g_netcdf_mutex;
#endif

// to deal with fortran fixed length strings
// which are not properly nulll terminated
static void crtrim(char *s, long n)
{
    if (!s || (n == 0)) return;
    char c = s[--n];
    while ((n > 0) && ((c == ' ') || (c == '\n') ||
        (c == '\t') || (c == '\r')))
    {
        s[n] = '\0';
        c = s[--n];
    }
}

// RAII for managing netcdf files
class netcdf_handle
{
public:
    netcdf_handle() : m_handle(0)
    {}

    // initialize with a handle returned from
    // nc_open/nc_create etc
    netcdf_handle(int h) : m_handle(h)
    {}

    // close the file during destruction
    ~netcdf_handle()
    { this->close(); }

    // this is a move only class, and should
    // only be initialized with an valid handle
    netcdf_handle(const netcdf_handle &) = delete;
    void operator=(const netcdf_handle &) = delete;

    // move construction takes ownership
    // from the other object
    netcdf_handle(netcdf_handle &&other)
    {
        m_handle = other.m_handle;
        other.m_handle = 0;
    }

    // move assignment takes ownership
    // from the other object
    void operator=(netcdf_handle &&other)
    {
        this->close();
        m_handle = other.m_handle;
        other.m_handle = 0;
    }

    // open the file
    int open(const std::string &file_path)
    {
        if (m_handle)
        {
            TECA_ERROR("Handle in use, close before re-opening")
            return -1;
        }

        int ierr = 0;
        std::lock_guard<std::mutex> lock(g_netcdf_mutex);
        if ((ierr = nc_open(file_path.c_str(), NC_NOWRITE, &m_handle)) != NC_NOERR)
        {
            TECA_ERROR("Failed to open " << file_path << ". " << nc_strerror(ierr))
            return -1;
        }

        return 0;
    }

    // close the file
    void close()
    {
        if (m_handle)
        {
            std::lock_guard<std::mutex> lock(g_netcdf_mutex);
            nc_close(m_handle);
            m_handle = 0;
        }
    }

    // returns a reference to the handle
    int &get()
    { return m_handle; }

private:
    int m_handle;
};

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

        netcdf_handle fh;
        if (fh.open(file_path))
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
        if (((ierr = nc_inq_dimid(file_id, m_variable.c_str(), &var_id)) != NC_NOERR)
            || ((ierr = nc_inq_dimlen(file_id, var_id, &var_size)) != NC_NOERR)
            || ((ierr = nc_inq_varid(file_id, m_variable.c_str(), &var_id)) != NC_NOERR)
            || ((ierr = nc_inq_vartype(file_id, var_id, &var_type)) != NC_NOERR))
        {
            TECA_ERROR("Failed to read metadata for variable \"" << m_variable
                << "\" from \"" << m_file << "\". " << nc_strerror(ierr))
            return std::make_pair(m_id, nullptr);
        }

        // allocate a buffer and read the variable.
        NC_DISPATCH_FP(var_type,
            size_t start = 0;
            p_teca_variant_array_impl<NC_T> var = teca_variant_array_impl<NC_T>::New();
            var->resize(var_size);
            if ((ierr = nc_get_vara(file_id, var_id, &start, &var_size, var->get())) != NC_NOERR)
            {
                TECA_ERROR("Failed to read variable \"" << m_variable  << "\" from \""
                    << m_file << "\". " << nc_strerror(ierr))
                return std::make_pair(m_id, nullptr);
            }
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
        TECA_POPTS_GET(std::string, prefix, files_regex,
            "a regular expression that matches the set of files "
            "comprising the dataset")
        TECA_POPTS_GET(std::string, prefix, file_name,
            "a single path/file name to read. may be used in place "
            "of files_regex")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "name of variable that has x axis coordinates (lon)")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "name of variable that has y axis coordinates (lat)")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "name of variable that has z axis coordinates ()")
        TECA_POPTS_GET(std::string, prefix, t_axis_variable,
            "name of variable that has t axis coordinates (time)")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "set the number of I/O threads (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, files_regex)
    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_axis_variable)
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
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
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

        if (!this->file_name.empty())
        {
            // use file name
            path = teca_file_util::path(this->file_name);
            files.push_back(teca_file_util::filename(this->file_name));
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
                metadata_cache_path[i] + PATH_SEP + metadata_cache_key;

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
            int file_id = 0;
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

#if !defined(HDF5_THREAD_SAFE)
            {std::lock_guard<std::mutex> lock(g_netcdf_mutex);
#endif
            if ((ierr = nc_open(file.c_str(), NC_NOWRITE, &file_id)) != NC_NOERR)
            {
                TECA_ERROR("Failed to open " << file << endl << nc_strerror(ierr))
                return teca_metadata();
            }

            // query mesh axes
            if (((ierr = nc_inq_dimid(file_id, x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(file_id, x_id, &n_x)) != NC_NOERR)
                || ((ierr = nc_inq_varid(file_id, x_axis_variable.c_str(), &x_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(file_id, x_id, &x_t)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query x axis variable \"" << x_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }

            if (!y_axis_variable.empty()
                && (((ierr = nc_inq_dimid(file_id, y_axis_variable.c_str(), &y_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(file_id, y_id, &n_y)) != NC_NOERR)
                || ((ierr = nc_inq_varid(file_id, y_axis_variable.c_str(), &y_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(file_id, y_id, &y_t)) != NC_NOERR)))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query y axis variable \"" << y_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }

            if (!z_axis_variable.empty()
                && (((ierr = nc_inq_dimid(file_id, z_axis_variable.c_str(), &z_id)) != NC_NOERR)
                || ((ierr = nc_inq_dimlen(file_id, z_id, &n_z)) != NC_NOERR)
                || ((ierr = nc_inq_varid(file_id, z_axis_variable.c_str(), &z_id)) != NC_NOERR)
                || ((ierr = nc_inq_vartype(file_id, z_id, &z_t)) != NC_NOERR)))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to query z axis variable \"" << z_axis_variable
                    << "\" in file \"" << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }

            // enumerate mesh arrays and their attributes
            if (((ierr = nc_inq_nvars(file_id, &n_vars)) != NC_NOERR))
            {
                this->clear_cached_metadata();
                TECA_ERROR(
                    << "Failed to get the number of variables in file \""
                    << file << "\"" << endl
                    << nc_strerror(ierr))
                return teca_metadata();
            }

            teca_metadata atrs;
            std::vector<std::string> vars;
            std::vector<std::string> time_vars; // anything that has the time dimension as it's only dim
            for (int i = 0; i < n_vars; ++i)
            {
                char var_name[NC_MAX_NAME + 1] = {'\0'};
                nc_type var_type = 0;
                int n_dims = 0;
                int dim_id[NC_MAX_VAR_DIMS] = {0};
                int n_atts = 0;

                if ((ierr = nc_inq_var(file_id, i, var_name,
                        &var_type, &n_dims, dim_id, &n_atts)) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_ERROR(
                        << "Failed to query " << i << "th variable, "
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }

                // skip scalars
                if (n_dims == 0)
                    continue;

                std::vector<size_t> dims;
                std::vector<std::string> dim_names;
                for (int ii = 0; ii < n_dims; ++ii)
                {
                    char dim_name[NC_MAX_NAME + 1] = {'\0'};
                    size_t dim = 0;
                    if ((ierr = nc_inq_dim(file_id, dim_id[ii], dim_name, &dim)) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
                            << "Failed to query " << ii << "th dimension of variable, "
                            << var_name << ", " << file << endl << nc_strerror(ierr))
                        return teca_metadata();
                    }

                    dim_names.push_back(dim_name);
                    dims.push_back(dim);
                }

                vars.push_back(var_name);

                if ((n_dims == 1) && (dim_names[0] == t_axis_variable))
                    time_vars.push_back(var_name);

                teca_metadata atts;
                atts.insert("id", i);
                atts.insert("dims", dims);
                atts.insert("dim_names", dim_names);
                atts.insert("type", var_type);
                atts.insert("centering", std::string("point"));

                for (int ii = 0; ii < n_atts; ++ii)
                {
                    char att_name[NC_MAX_NAME + 1] = {'\0'};
                    nc_type att_type = 0;
                    size_t att_len = 0;
                    if (((ierr = nc_inq_attname(file_id, i, ii, att_name)) != NC_NOERR)
                        || ((ierr = nc_inq_att(file_id, i, att_name, &att_type, &att_len)) != NC_NOERR))
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
                            << "Failed to query " << ii << "th attribute of variable, "
                            << var_name << ", " << file << endl << nc_strerror(ierr))
                        return teca_metadata();
                    }
                    if (att_type == NC_CHAR)
                    {
                        char *buffer = static_cast<char*>(malloc(att_len + 1));
                        buffer[att_len] = '\0';
                        nc_get_att_text(file_id, i, att_name, buffer);
                        crtrim(buffer, att_len);
                        atts.insert(att_name, std::string(buffer));
                        free(buffer);
                    }
                    else
                    {
                        NC_DISPATCH(att_type,
                          NC_T *buffer = static_cast<NC_T*>(malloc(att_len));
                          nc_get_att(file_id, i, att_name, buffer);
                          atts.insert(att_name, buffer, att_len);
                          free(buffer);
                          )
                    }
                }

                atrs.insert(var_name, atts);
            }

            this->internals->metadata.insert("variables", vars);
            this->internals->metadata.insert("attributes", atrs);
            this->internals->metadata.insert("time variables", time_vars);

            // read spatial coordinate arrays
            NC_DISPATCH_FP(x_t,
                size_t x_0 = 0;
                p_teca_variant_array_impl<NC_T> x = teca_variant_array_impl<NC_T>::New(n_x);
                if ((ierr = nc_get_vara(file_id, x_id, &x_0, &n_x, x->get())) != NC_NOERR)
                {
                    this->clear_cached_metadata();
                    TECA_ERROR(
                        << "Failed to read x axis, " << x_axis_variable << endl
                        << file << endl << nc_strerror(ierr))
                    return teca_metadata();
                }
                x_axis = x;
                )

            if (!y_axis_variable.empty())
            {
                NC_DISPATCH_FP(y_t,
                    size_t y_0 = 0;
                    p_teca_variant_array_impl<NC_T> y = teca_variant_array_impl<NC_T>::New(n_y);
                    if ((ierr = nc_get_vara(file_id, y_id, &y_0, &n_y, y->get())) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
                            << "Failed to read y axis, " << y_axis_variable << endl
                            << file << endl << nc_strerror(ierr))
                        return teca_metadata();
                    }
                    y_axis = y;
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
                    if ((ierr = nc_get_vara(file_id, z_id, &z_0, &n_z, z->get())) != NC_NOERR)
                    {
                        this->clear_cached_metadata();
                        TECA_ERROR(
                            << "Failed to read z axis, " << z_axis_variable << endl
                            << file << endl << nc_strerror(ierr))
                        return teca_metadata();
                    }
                    z_axis = z;
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
#if !defined(HDF5_THREAD_SAFE)
            }
#endif

            // collect time steps from this and the rest of the files.
            // there are a couple of  performance issues on Lustre.
            // 1) opening a file is slow, there's latency due to contentions
            // 2) reading the time axis is very slow as it's not stored
            //    contiguously by convention. ie. time is an "unlimted"
            //    NetCDF dimension.
            // when procesing large numbers of files these issues kill
            // serial performance. hence we are reading time dimension
            // in parallel.
            read_variable_queue_t thread_pool(this->thread_pool_size,
                true, true, false);

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
            else
            {
                // make a dummy time axis, this enables parallelization over file sets
                // that do not have time dimension. However, there is no guarantee on the
                // order of the dummy axis to the lexical ordering of the files and there
                // will be no calendaring information. As a result many time aware algorithms
                // will not work.
                size_t n_files = files.size();
                NC_DISPATCH_FP(x_t,
                    p_teca_variant_array_impl<NC_T> t = teca_variant_array_impl<NC_T>::New(n_files);
                    for (size_t i = 0; i < n_files; ++i)
                    {
                        t->set(i, NC_T(i));
                        step_count.push_back(1);
                    }
                    t_axis = t;
                    )
            }

            teca_metadata coords;
            coords.insert("x_variable", x_axis_variable);
            coords.insert("y_variable", (z_axis_variable.empty() ? "y" : z_axis_variable));
            coords.insert("z_variable", (z_axis_variable.empty() ? "z" : z_axis_variable));
            coords.insert("t_variable", (z_axis_variable.empty() ? "t" : z_axis_variable));
            coords.insert("x", x_axis);
            coords.insert("y", y_axis);
            coords.insert("z", z_axis);
            coords.insert("t", t_axis);

            std::vector<size_t> whole_extent(6, 0);
            whole_extent[1] = n_x - 1;
            whole_extent[3] = n_y - 1;
            whole_extent[5] = n_z - 1;
            this->internals->metadata.insert("whole_extent", whole_extent);
            this->internals->metadata.insert("coordinates", coords);
            this->internals->metadata.insert("files", files);
            this->internals->metadata.insert("root", path);
            this->internals->metadata.insert("step_count", step_count);
            this->internals->metadata.insert("number_of_time_steps", t_axis->size());

            this->internals->metadata.to_stream(stream);

#if defined(TECA_HAS_OPENSSL)
            // cache metadata on disk
            bool cached_metadata = false;
            for (int i = 0; i < 3; ++i)
            {
                std::string metadata_cache_file =
                    metadata_cache_path[i] + PATH_SEP + metadata_cache_key;

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
            stream.broadcast(root_rank);
#endif
    }
#if defined(TECA_HAS_MPI)
    else
    if (is_init)
    {
        // all other ranks receive the metadata from the root
        stream.broadcast(root_rank);

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
    netcdf_handle fh;
    if (fh.open(file_path))
    {
        TECA_ERROR("time_step=" << time_step << " Failed to open \"" << file << "\"")
        return nullptr;
    }
    int file_id = fh.get();

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates(out_x);
    mesh->set_y_coordinates(out_y);
    mesh->set_z_coordinates(out_z);
    mesh->set_time(t);
    mesh->set_time_step(time_step);
    mesh->set_whole_extent(whole_extent);
    mesh->set_extent(extent);

    // get the time offset
    teca_metadata atrs;
    if (this->internals->metadata.get("attributes", atrs))
    {
        TECA_ERROR("time_step=" << time_step
            << " metadata missing \"attributes\"")
        return nullptr;
    }

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
    size_t n_arrays = arrays.size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get metadata
        teca_metadata atts;
        int type = 0;
        int id = 0;
        p_teca_string_array dim_names;

        if (atrs.get(arrays[i], atts)
            || atts.get("type", 0, type)
            || atts.get("id", 0, id)
            || !(dim_names = std::dynamic_pointer_cast<teca_string_array>(atts.get("dim_names"))))
        {
            TECA_ERROR("metadata issue can't read \"" << arrays[i] << "\"")
            continue;
        }

        // check if it's a mesh variable
        bool mesh_var = false;
        size_t n_dims = dim_names->size();
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
        if (!mesh_var)
        {
            TECA_ERROR("time_step=" << time_step
                << " dimension mismatch. \"" << arrays[i]
                << "\" is not a mesh variable")
            continue;
        }

        // read
        p_teca_variant_array array;
        NC_DISPATCH(type,
            p_teca_variant_array_impl<NC_T> a = teca_variant_array_impl<NC_T>::New(mesh_size);
            if ((ierr = nc_get_vara(file_id,  id, &starts[0], &counts[0], a->get())) != NC_NOERR)
            {
                TECA_ERROR("time_step=" << time_step
                    << " Failed to read variable \"" << arrays[i] << "\" "
                    << file << endl << nc_strerror(ierr))
                continue;
            }
            array = a;
            )
        mesh->get_point_arrays()->append(arrays[i], array);
    }

    // read time vars
    std::vector<std::string> time_vars;
    this->internals->metadata.get("time variables", time_vars);
    size_t n_time_vars = time_vars.size();
    for (size_t i = 0; i < n_time_vars; ++i)
    {
        // get metadata
        teca_metadata atts;
        int type = 0;
        int id = 0;

        if (atrs.get(time_vars[i], atts)
            || atts.get("type", 0, type)
            || atts.get("id", 0, id))
        {
            TECA_ERROR("time_step=" << time_step
                << " metadata issue can't read \"" << time_vars[i] << "\"")
            continue;
        }

        // read
        int ierr = 0;
        p_teca_variant_array array;
        size_t one = 1;
        NC_DISPATCH(type,
            p_teca_variant_array_impl<NC_T> a = teca_variant_array_impl<NC_T>::New(1);
            if ((ierr = nc_get_vara(file_id,  id, &starts[0], &one, a->get())) != NC_NOERR)
            {
                TECA_ERROR("time_step=" << time_step
                    << " Failed to read \"" << time_vars[i] << "\" "
                    << file << endl << nc_strerror(ierr))
                continue;
            }
            array = a;
            )
        mesh->get_information_arrays()->append(time_vars[i], array);
    }

    return mesh;
}
