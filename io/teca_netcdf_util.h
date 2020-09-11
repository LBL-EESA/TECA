#ifndef teca_netcdf_util_h
#define teca_netcdf_util_h

#include "teca_config.h"
#include "teca_mpi.h"
#include "teca_metadata.h"

#include <mutex>
#include <string>

#include <netcdf.h>
#if defined(TECA_HAS_NETCDF_MPI)
#include <netcdf_par.h>
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
        TECA_ERROR("netcdf type code " << tc_               \
            << " is not supported")                         \
    }

#define NC_DISPATCH_CASE(cc_, tt_, code_)   \
    case cc_:                               \
    {                                       \
        using NC_T = tt_;                   \
        code_                               \
        break;                              \
    }

namespace teca_netcdf_util
{

// traits class mapping to/from netcdf
template<typename num_t> class netcdf_tt {};
template<int nc_enum> class cpp_tt {};

#define DECLARE_NETCDF_TT(cpp_t_, nc_c_) \
template <> class netcdf_tt<cpp_t_>      \
{ public: enum { type_code = nc_c_ }; };
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

#define DECLARE_CPP_TT(cpp_t_, nc_c_) \
template <> class cpp_tt<nc_c_>       \
{ public: using type = cpp_t_; };
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

// to deal with fortran fixed length strings
// which are not properly nulll terminated
void crtrim(char *s, long n);

// NetCDF 3 is not threadsafe. The HDF5 C-API can be compiled to be threadsafe,
// but it is usually not. NetCDF uses HDF5-HL API to access HDF5, but HDF5-HL
// API is not threadsafe without the --enable-unsupported flag. For all those
// reasons it's best for the time being to protect all NetCDF I/O.
std::mutex &get_netcdf_mutex();

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

    // open the file. this can be used from MPI parallel runs, but collective
    // I/O is not possible when a file is opend this way. Returns 0 on success.
    int open(const std::string &file_path, int mode);

    // open the file. this can be used when collective I/O is desired. the
    // passed in communcator specifies the subset of ranks that will access
    // the file. Calling this when linked to a non-MPI enabled NetCDF install,
    // from a parallel run will, result in an error. Returns 0 on success.
    int open(MPI_Comm comm, const std::string &file_path, int mode);

    // create the file. this can be used from MPI parallel runs, but collective
    // I/O is not possible when a file is created this way. Returns 0 on success.
    int create(const std::string &file_path, int mode);

    // create the file. this can be used when collective I/O is desired. the
    // passed in communcator specifies the subset of ranks that will access
    // the file. Calling this when linked to a non-MPI enabled NetCDF install,
    // from a parallel run will, result in an error. Returns 0 on success.
    int create(MPI_Comm comm, const std::string &file_path, int mode);

    // close the file
    int close();

    // flush all data to disk
    int flush();

    // returns a reference to the handle
    int &get()
    { return m_handle; }

    // test if the handle is valid
    operator bool() const
    { return m_handle > 0; }

private:
    int m_handle;
};

// read the specified variable attribute by name.
// it's value is stored in the metadata object
// return is non-zero if an error occurred
int read_attribute(netcdf_handle &fh, int var_id,
    const std::string &att_name, teca_metadata &atts);

// read the specified variable attribute by id
// it's value is stored in the metadata object
// return is non-zero if an error occurred
int read_attribute(netcdf_handle &fh, int var_id,
    int att_id, teca_metadata &atts);

// read the specified variable's name, dimensions, and it's associated
// NetCDF attributes into the metadata object. Additonally the following
// key/value pairs are added and useful for subsequent I/O and processing
//
//  cf_id - the NetCDF variable id that can be used to read the variable
//  cf_dims - a vector of the NetCDF dimension lengths (i.e. the variable's shape)
//  cf_dim_names - a vector of the names of the NetCDF dimensions
//  cf_type_code - the NetCDF type code
//  type_code - the TECA type code
//  centering - for now it is set to teca_array_attributes::point_centering
//
// return is non-zero if an error occurred
int read_variable_attributes(netcdf_handle &fh, int var_id,
    std::string &name, teca_metadata &atts);

int read_variable_attributes(netcdf_handle &fh,
    const std::string &name, teca_metadata &atts);
}

#endif
