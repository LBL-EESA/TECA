#ifndef teca_netcdf_util_h
#define teca_netcdf_util_h

/// @file

#include "teca_config.h"
#include "teca_mpi.h"
#include "teca_metadata.h"
#include "teca_cpu_thread_pool.h"

#include <mutex>
#include <string>

#include <netcdf.h>
#if defined(TECA_HAS_NETCDF_MPI)
#include <netcdf_par.h>
#endif

/// macro to help with netcdf floating point data types
#define NC_DISPATCH_FP(tc_, ...)                                \
    switch (tc_)                                                \
    {                                                           \
    NC_DISPATCH_CASE(NC_FLOAT, float, __VA_ARGS__)              \
    NC_DISPATCH_CASE(NC_DOUBLE, double, __VA_ARGS__)            \
    default:                                                    \
        TECA_ERROR("netcdf type __VA_ARGS__ " << tc_            \
            << " is not a floating point type")                 \
    }

/// macro to help with netcdf data types
#define NC_DISPATCH(tc_, ...)                                     \
    switch (tc_)                                                  \
    {                                                             \
    NC_DISPATCH_CASE(NC_BYTE, char, __VA_ARGS__)                  \
    NC_DISPATCH_CASE(NC_UBYTE, unsigned char, __VA_ARGS__)        \
    NC_DISPATCH_CASE(NC_CHAR, char, __VA_ARGS__)                  \
    NC_DISPATCH_CASE(NC_SHORT, short int, __VA_ARGS__)            \
    NC_DISPATCH_CASE(NC_USHORT, unsigned short int, __VA_ARGS__)  \
    NC_DISPATCH_CASE(NC_INT, int, __VA_ARGS__)                    \
    NC_DISPATCH_CASE(NC_UINT, unsigned int, __VA_ARGS__)          \
    NC_DISPATCH_CASE(NC_INT64, long long, __VA_ARGS__)            \
    NC_DISPATCH_CASE(NC_UINT64, unsigned long long, __VA_ARGS__)  \
    NC_DISPATCH_CASE(NC_FLOAT, float, __VA_ARGS__)                \
    NC_DISPATCH_CASE(NC_DOUBLE, double, __VA_ARGS__)              \
    default:                                                      \
        TECA_ERROR("netcdf type code " << tc_                     \
            << " is not supported")                               \
    }

/// macro that executes code when the type code is matched.
#define NC_DISPATCH_CASE(cc_, tt_, ...)                                     \
    case cc_:                                                               \
    {                                                                       \
        using NC_NT = tt_;                                                  \
        using NC_TT = teca_variant_array_impl<tt_>;                         \
        using NC_CTT = const teca_variant_array_impl<tt_>;                  \
        using NC_PT = std::shared_ptr<teca_variant_array_impl<tt_>>;        \
        using NC_CPT = std::shared_ptr<const teca_variant_array_impl<tt_>>; \
        using NC_SP = std::shared_ptr<tt_>;                                 \
        using NC_CSP = std::shared_ptr<const tt_>;                          \
        __VA_ARGS__                                                         \
        break;                                                              \
    }

/// Codes dealing with NetCDF I/O calls
namespace teca_netcdf_util
{

/// A traits class mapping to netcdf from C++
template<typename num_t> class TECA_EXPORT netcdf_tt {};

/// A traits class mapping to C++ from netcdf
template<int nc_enum> class TECA_EXPORT cpp_tt {};

#define DECLARE_NETCDF_TT(cpp_t_, nc_c_)                                 \
/** A traits class mapping to NetCDF from C++, specialized for cpp_t_ */ \
template <> class netcdf_tt<cpp_t_>                                      \
{                                                                        \
public:                                                                  \
    enum { type_code = nc_c_ };                                          \
    static const char *name() { return #nc_c_; }                         \
};
DECLARE_NETCDF_TT(char, NC_BYTE)
DECLARE_NETCDF_TT(unsigned char, NC_UBYTE)
//DECLARE_NETCDF_TT(char, NC_CHAR)
DECLARE_NETCDF_TT(short int, NC_SHORT)
DECLARE_NETCDF_TT(unsigned short int, NC_USHORT)
DECLARE_NETCDF_TT(int, NC_INT)
DECLARE_NETCDF_TT(long, NC_LONG)
DECLARE_NETCDF_TT(unsigned long, NC_LONG)
DECLARE_NETCDF_TT(unsigned int, NC_UINT)
DECLARE_NETCDF_TT(long long, NC_INT64)
DECLARE_NETCDF_TT(unsigned long long, NC_UINT64)
DECLARE_NETCDF_TT(float, NC_FLOAT)
DECLARE_NETCDF_TT(double, NC_DOUBLE)

#define DECLARE_CPP_TT(cpp_t_, nc_c_)                                    \
/** A traits class mapping to C++ from NetCDF, specialized for cpp_t_ */ \
template <> class cpp_tt<nc_c_>                                          \
{                                                                        \
public:                                                                  \
    using type = cpp_t_;                                                 \
    static const char *name() { return #cpp_t_; }                        \
};
DECLARE_CPP_TT(char, NC_BYTE)
DECLARE_CPP_TT(unsigned char, NC_UBYTE)
//DECLARE_CPP_TT(char, NC_CHAR)
DECLARE_CPP_TT(short int, NC_SHORT)
DECLARE_CPP_TT(unsigned short int, NC_USHORT)
DECLARE_CPP_TT(int, NC_INT)
//DECLARE_CPP_TT(long, NC_LONG)
//DECLARE_CPP_TT(unsigned long, NC_LONG)
DECLARE_CPP_TT(unsigned int, NC_UINT)
DECLARE_CPP_TT(long long, NC_INT64)
DECLARE_CPP_TT(unsigned long long, NC_UINT64)
DECLARE_CPP_TT(float, NC_FLOAT)
DECLARE_CPP_TT(double, NC_DOUBLE)

/** To deal with fortran fixed length strings which are not properly nulll
 * terminated.
 */
void crtrim(char *s, long n);

/** NetCDF 3 is not threadsafe. The HDF5 C-API can be compiled to be
 * threadsafe, but it is usually not. NetCDF uses HDF5-HL API to access HDF5,
 * but HDF5-HL API is not threadsafe without the --enable-unsupported flag. For
 * all those reasons it's best for the time being to protect all NetCDF I/O.
 */
std::mutex &get_netcdf_mutex();

/// A RAII class for managing NETCDF files. The file is kept open while the object exists.
class TECA_EXPORT netcdf_handle
{
public:
    netcdf_handle() : m_handle(0)
    {}

    /** Initialize with a handle returned from nc_open/nc_create etc. */
    netcdf_handle(int h) : m_handle(h)
    {}

    /** Close the file during destruction. */
    ~netcdf_handle()
    { this->close(); }

    /**
     * This is a move only class, and should
     * only be initialized with an valid handle.
     */
    netcdf_handle(const netcdf_handle &) = delete;
    void operator=(const netcdf_handle &) = delete;

    /** Move construction takes ownership from the other object. */
    netcdf_handle(netcdf_handle &&other)
    {
        m_handle = other.m_handle;
        other.m_handle = 0;
    }

    /** Move assignment takes ownership from the other object. */
    void operator=(netcdf_handle &&other)
    {
        this->close();
        m_handle = other.m_handle;
        other.m_handle = 0;
    }

    /**
     * Open the file. this can be used from MPI parallel runs, but collective
     * I/O is not possible when a file is opened this way. Returns 0 on
     * success.
     */
    int open(const std::string &file_path, int mode);

    /**
     * Open the file. this can be used when collective I/O is desired. the
     * passed in communicator specifies the subset of ranks that will access
     * the file. Calling this when linked to a non-MPI enabled NetCDF install,
     * from a parallel run will, result in an error. Returns 0 on success.
     */
    int open(MPI_Comm comm, const std::string &file_path, int mode);

    /**
     * Create the file. this can be used from MPI parallel runs, but collective
     * I/O is not possible when a file is created this way. Returns 0 on
     * success.
     */
    int create(const std::string &file_path, int mode);

    /**
     * Create the file. this can be used when collective I/O is desired. the
     * passed in communicator specifies the subset of ranks that will access
     * the file. Calling this when linked to a non-MPI enabled NetCDF install,
     * from a parallel run will, result in an error. Returns 0 on success.
     */
    int create(MPI_Comm comm, const std::string &file_path, int mode);

    /** Close the file. */
    int close();

    /** Flush all data to disk. */
    int flush();

    /** Returns a reference to the handle. */
    int &get()
    { return m_handle; }

    /** Test if the handle is valid. */
    operator bool() const
    { return m_handle > 0; }

private:
    int m_handle;
};

/**
 * Read the specified variable attribute by name.
 * Its value is stored in the metadata object
 * return is non-zero if an error occurred.
 */
TECA_EXPORT
int read_attribute(netcdf_handle &fh, int var_id,
    const std::string &att_name, teca_metadata &atts);

/**
 * Read the specified variable attribute by id.
 * Its value is stored in the metadata object
 * return is non-zero if an error occurred.
 */
TECA_EXPORT
int read_attribute(netcdf_handle &fh, int var_id,
    int att_id, teca_metadata &atts);

/**
 * Read the specified variable's name, dimensions, and it's associated
 * NetCDF attributes into the metadata object. Additionally the following
 * key/value pairs are added and useful for subsequent I/O and processing
 *
 * <H4 ID="cf_atts">CF Attributes</H4>
 *
 *  | Key             | Description                                          |
 *  | ----            | -----------                                          |
 *  | cf_id           | The NetCDF variable id that can be used to read the  |
 *  |                 | variable.                                            |
 *  | cf_dims         | A vector of the NetCDF dimension lengths (i.e. the   |
 *  |                 | variable's shape).                                   |
 *  | cf_dim_names    | A vector of the names of the NetCDF dimensions.      |
 *  | cf_type_code    | The NetCDF type code.                                |
 *  | type_code       | The teca_variant_array::code type code.              |
 *  | centering       | The mesh centering, point_centering or no_centering  |
 *  | have_mesh_dim   | Flags indicating the presence of the x,y,z, and t    |
 *  |                 | mesh dimensions                                      |
 *  | mesh_dim_active | Flags indicating if the x,y,z, and t dimension is    |
 *  |                 | active.                                              |
 *
 * In order for centering and have_mesh_dim flags to be set, the x_variable,
 * y_variable, z_variable, and t_variable must be specified.
 *
 * If dimension is 1 and clamp_dimensions_of_one is set then the dimension is
 * marked as inactive.
 *
 * returns non-zero if an error occurred.
 */
TECA_EXPORT
int read_variable_attributes(netcdf_handle &fh, int var_id,
    const std::string &x_variable, const std::string &y_variable,
    const std::string &z_variable, const std::string &t_variable,
    int clamp_dimensions_of_one, std::string &name, teca_metadata &atts);

/**
 * Read the specified variable's name, dimensions, and it's associated
 * NetCDF attributes into the metadata object. See <A HREF="#cf_atts">CF Attributes</A>
 * for details of attributes returned. returns non-zero if an error occurred.
 */
TECA_EXPORT
int read_variable_attributes(netcdf_handle &fh, int var_id,
    std::string &name, teca_metadata &atts);

/**
 * Read the specified variable's dimensions, and it's associated
 * NetCDF attributes into the metadata object. See <A HREF="#cf_atts">CF Attributes</A>
 * for details of attributes returned. returns non-zero if an error occurred.
 */
TECA_EXPORT
int read_variable_attributes(netcdf_handle &fh,
    const std::string &name,
    const std::string &x_variable, const std::string &y_variable,
    const std::string &z_variable, const std::string &t_variable,
    int clamp_dimensions_of_one, teca_metadata &atts);

/**
 * Read the specified variable's dimensions, and it's associated
 * NetCDF attributes into the metadata object. See <A HREF="#cf_atts">CF Attributes</A>
 * for details of attributes returned. returns non-zero if an error occurred.
 */
TECA_EXPORT
int read_variable_attributes(netcdf_handle &fh,
    const std::string &var_name, teca_metadata &atts);

/// Functional that reads and returns a variable from the named file.
/**
 * We're doing this so we can do thread
 * parallel I/O to hide some of the cost of opening files
 * on Lustre and to hide the cost of reading time coordinate
 * which is typically very expensive as NetCDF stores
 * unlimited dimensions non-contiguously.
 *
 * @note
 * Thu 09 Apr 2020 05:45:29 AM PDT
 * Threading these operations worked well in NetCDF 3, however
 * in NetCDF 4 backed by HDF5 necessary locking eliminates any
 * speed up.
 */
class TECA_EXPORT read_variable_and_attributes
{
public:
    /** Data and task types. */
    using data_elem_t = std::pair<p_teca_variant_array, teca_metadata>;
    using data_t = std::pair<unsigned long, data_elem_t>;
    using task_t = std::packaged_task<data_t()>;
    using queue_t = teca_cpu_thread_pool<task_t, data_t>;
    using p_queue_t = std::shared_ptr<queue_t>;

    read_variable_and_attributes(const std::string &path, const std::string &file,
        unsigned long id, const std::string &variable) : m_path(path),
        m_file(file), m_variable(variable), m_id(id)
    {}

    static
    data_t package(unsigned long id,
        p_teca_variant_array var = nullptr,
        const teca_metadata &md = teca_metadata())
    {
        return std::make_pair(id, std::make_pair(var, md));
    }

    data_t operator()(int device_id = -1);

private:
    std::string m_path;
    std::string m_file;
    std::string m_variable;
    unsigned long m_id;
};

/// Function that reads and returns a variable from the named file.
/**
 * we're doing this so we can do thread
 * parallel I/O to hide some of the cost of opening files
 * on Lustre and to hide the cost of reading time coordinate
 * which is typically very expensive as NetCDF stores
 * unlimited dimensions non-contiguously
 *
 * @note
 * Thu 09 Apr 2020 05:45:29 AM PDT
 * Threading these operations worked well in NetCDF 3, however
 * in NetCDF 4 backed by HDF5 necessary locking eliminates any
 * speed up.
 */
class TECA_EXPORT read_variable
{
public:
    /** Data and task types. */
    using data_t = std::pair<unsigned long, p_teca_variant_array>;
    using task_t = std::packaged_task<data_t(int)>;
    using queue_t = teca_cpu_thread_pool<task_t, data_t>;
    using p_queue_t = std::shared_ptr<queue_t>;


    read_variable(const std::string &path, const std::string &file,
        unsigned long id, const std::string &variable) : m_path(path),
        m_file(file), m_variable(variable), m_id(id)
    {}

    static
    data_t package(unsigned long id,
        p_teca_variant_array var = nullptr)
    {
        return std::make_pair(id, var);
    }

    data_t operator()(int device_id = -1);

private:
    std::string m_path;
    std::string m_file;
    std::string m_variable;
    unsigned long m_id;
};


/**
 * Write the attributes in array_atts to the variable identified by var_id the
 * name is used in error messages. Returns zero of successful.
 */
TECA_EXPORT
int write_variable_attributes(netcdf_handle &fh, int var_id,
    teca_metadata &array_atts);

}
#endif
