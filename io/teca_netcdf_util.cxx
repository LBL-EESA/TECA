#include "teca_netcdf_util.h"
#include "teca_common.h"

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
