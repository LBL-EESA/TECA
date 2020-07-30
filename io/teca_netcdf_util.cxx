#include "teca_netcdf_util.h"
#include "teca_common.h"
#include "teca_system_interface.h"
#include "teca_file_util.h"

#include <cstring>

static std::mutex g_netcdf_mutex;

namespace teca_netcdf_util
{

// **************************************************************************
void crtrim(char *s, long n)
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

// **************************************************************************
std::mutex &get_netcdf_mutex()
{
    return g_netcdf_mutex;
}

// --------------------------------------------------------------------------
int netcdf_handle::open(const std::string &file_path, int mode)
{
    if (m_handle)
    {
        TECA_ERROR("Handle in use, close before re-opening")
        return -1;
    }

    int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
     std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if ((ierr = nc_open(file_path.c_str(), mode, &m_handle)) != NC_NOERR)
    {
        TECA_ERROR("Failed to open \"" << file_path << "\". " << nc_strerror(ierr))
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int netcdf_handle::open(MPI_Comm comm, const std::string &file_path, int mode)
{
#if !defined(TECA_HAS_NETCDF_MPI)
#if defined(TECA_HAS_MPI)
    int n_ranks = 1;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_size(comm, &n_ranks);
    if (n_ranks > 1)
    {
        // it would open all kinds of confusion and chaos to let this call
        // succeed without collective I/O capabilities in a MPI parallel run
        // error out now.
        TECA_ERROR("Collective I/O attempted with a non-MPI NetCDF install")
        return -1;
    }
    // forward to the non-collective library call
    return this->open(file_path, mode);
#else
    // forward to the non-collective library call
    return this->open(file_path, mode);
#endif
#else
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (!is_init)
    {
        // forward to the non-collective library call
        return this->open(file_path, mode);
    }

    // open the file for collective parallel i/o
    if (m_handle)
    {
        TECA_ERROR("Handle in use, close before re-opening")
        return -1;
    }

    int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
     std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if ((ierr = nc_open_par(file_path.c_str(), mode,
        comm, MPI_INFO_NULL, &m_handle)) != NC_NOERR)
    {
        TECA_ERROR("Failed to open \"" << file_path << "\". " << nc_strerror(ierr))
        return -1;
    }

    return 0;
#endif
}

// --------------------------------------------------------------------------
int netcdf_handle::create(const std::string &file_path, int mode)
{
    if (m_handle)
    {
        TECA_ERROR("Handle in use, close before re-opening")
        return -1;
    }

    int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
     std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if ((ierr = nc_create(file_path.c_str(), mode, &m_handle)) != NC_NOERR)
    {
        TECA_ERROR("Failed to create \"" << file_path << "\". " << nc_strerror(ierr))
        return -1;
    }

    // add some global metadata for provenance
    if ((ierr = nc_put_att_text(m_handle, NC_GLOBAL, "TECA_VERSION_DESCR",
        strlen(TECA_VERSION_DESCR), TECA_VERSION_DESCR)))
    {
        TECA_ERROR("Failed to set version attribute." << nc_strerror(ierr))
        return -1;
    }

    std::string app_name =
        teca_file_util::filename(teca_system_interface::get_program_name());

    if (!app_name.empty() && (ierr = nc_put_att_text(m_handle, NC_GLOBAL,
        "APP_NAME", app_name.size(), app_name.c_str())))
    {
        TECA_ERROR("Failed to set app name attribute." << nc_strerror(ierr))
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int netcdf_handle::create(MPI_Comm comm, const std::string &file_path, int mode)
{
#if !defined(TECA_HAS_NETCDF_MPI)
#if defined(TECA_HAS_MPI)
    int n_ranks = 1;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_size(comm, &n_ranks);
    if (n_ranks > 1)
    {
        // it would create all kinds of confusion and chaos to let this call
        // succeed without collective I/O capabilities in a MPI parallel run
        // error out now.
        TECA_ERROR("Collective I/O attempted with a non-MPI NetCDF install. "
            "Re-install the NetCDF C library with --enable-parallel-4. See "
            "also the output of \"nc-config --has-parallel\" and/or the "
            "value of the \"NC-4 Parallel Support\" field in the file "
            "\"<prefix>/lib/libnetcdf.settings\" to determine if the install "
            "supports parallel I/O." )
        return -1;
    }
    // forward to the non-collective library call
    return this->create(file_path, mode);
#else
    // forward to the non-collective library call
    return this->create(file_path, mode);
#endif
#else
    // forward to the non-collective library call if MPI is not in use
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (!is_init)
    {
        return this->create(file_path, mode);
    }

    // create the file for collective parallel i/o
    if (m_handle)
    {
        TECA_ERROR("Handle in use, close before re-opening")
        return -1;
    }

    int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
     std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if ((ierr = nc_create_par(file_path.c_str(), mode,
        comm, MPI_INFO_NULL, &m_handle)) != NC_NOERR)
    {
        TECA_ERROR("Failed to create \"" << file_path << "\". " << nc_strerror(ierr))
        return -1;
    }

    // add some global metadata for provenance
    if ((ierr = nc_put_att_text(m_handle, NC_GLOBAL, "TECA_version",
        strlen(TECA_VERSION_DESCR), TECA_VERSION_DESCR)))
    {
        TECA_ERROR("Failed to set version attribute." << nc_strerror(ierr))
        return -1;
    }

    std::string app_name =
        teca_file_util::filename(teca_system_interface::get_program_name());

    if (!app_name.empty() && (ierr = nc_put_att_text(m_handle, NC_GLOBAL,
        "TECA_app_name", app_name.size(), app_name.c_str())))
    {
        TECA_ERROR("Failed to set app name attribute." << nc_strerror(ierr))
        return -1;
    }

    return 0;
#endif
}

// --------------------------------------------------------------------------
int netcdf_handle::flush()
{
#if !defined(HDF5_THREAD_SAFE)
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    int ierr = 0;
    if ((ierr = nc_sync(m_handle)) != NC_NOERR)
    {
        TECA_ERROR("Failed to sync file. " << nc_strerror(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int netcdf_handle::close()
{
    if (m_handle)
    {
#if !defined(HDF5_THREAD_SAFE)
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        nc_close(m_handle);
        m_handle = 0;
    }
    return 0;
}

}
