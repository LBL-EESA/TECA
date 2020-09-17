#include "teca_netcdf_util.h"
#include "teca_common.h"
#include "teca_system_interface.h"
#include "teca_file_util.h"
#include "teca_variant_array.h"
#include "teca_array_attributes.h"

#include <cstring>
#include <vector>

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

// **************************************************************************
int read_attribute(netcdf_handle &fh, int var_id,
    const std::string &att_name, teca_metadata &atts)
{
    int ierr = 0;
    int att_id = 0;
    if ((ierr = nc_inq_attid(fh.get(), var_id, att_name.c_str(),
        &att_id)) != NC_NOERR)
    {
        TECA_ERROR("Failed to get the id of attribute \""
            << att_name << "\" of variable " << var_id << std::endl
            << nc_strerror(ierr))
        return -1;
    }

    return teca_netcdf_util::read_attribute(fh, var_id, att_id, atts);
}

// **************************************************************************
int read_attribute(netcdf_handle &fh, int var_id, int att_id, teca_metadata &atts)
{
    int ierr = 0;
    char att_name[NC_MAX_NAME + 1] = {'\0'};
    nc_type att_type = 0;
    size_t att_len = 0;
#if !defined(HDF5_THREAD_SAFE)
    {
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if (((ierr = nc_inq_attname(fh.get(), var_id, att_id, att_name)) != NC_NOERR)
        || ((ierr = nc_inq_att(fh.get(), var_id, att_name, &att_type, &att_len)) != NC_NOERR))
    {
        TECA_ERROR("Failed to query the " << att_id << "th attribute of variable "
            << var_id << std::endl << nc_strerror(ierr))
        return -1;
    }
#if !defined(HDF5_THREAD_SAFE)
    }
#endif
    if (att_type == NC_CHAR)
    {
        char *tmp = static_cast<char*>(malloc(att_len + 1));
        tmp[att_len] = '\0';
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_get_att_text(fh.get(), var_id, att_name, tmp)) != NC_NOERR)
        {
            free(tmp);
            TECA_ERROR("Failed get text from the " << att_id << "th attribute \""
                << att_name << "\" of variable " << var_id << std::endl
                << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        teca_netcdf_util::crtrim(tmp, att_len);
        atts.set(att_name, std::string(tmp));

        free(tmp);

        return 0;
    }
    else if (att_type == NC_STRING)
    {
        char *strs[1] = {nullptr};
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_get_att_string(fh.get(), var_id, att_name, strs)) != NC_NOERR)
        {
            TECA_ERROR("Failed get string from the " << att_id << "th attribute \""
                << att_name << "\" of variable " << var_id << std::endl
                << nc_strerror(ierr))
            return -1;
        }
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
        return 0;
    }
    else
    {
        NC_DISPATCH(att_type,
            NC_T *tmp = static_cast<NC_T*>(malloc(sizeof(NC_T)*att_len));
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_get_att(fh.get(), var_id, att_name, tmp)) != NC_NOERR)
            {
                TECA_ERROR("Failed get the " << att_id << "th attribute \""
                    << att_name << "\" of variable " << var_id << std::endl
                    << nc_strerror(ierr))
                free(tmp);

            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            atts.set(att_name, tmp, att_len);
            free(tmp);

            return 0;
            )
    }

    TECA_ERROR("Failed to read the " << att_id << "th attribute of variable "
        << var_id << ". Unhandled case")
    return -1;
}

// **************************************************************************
int read_variable_attributes(netcdf_handle &fh, const std::string &var_name,
    teca_metadata &atts)
{
    int ierr = 0;
    int var_id = 0;
    int var_type = 0;
    nc_type var_nc_type = 0;
    int n_dims = 0;
    int dim_id[NC_MAX_VAR_DIMS] = {0};
    int n_atts = 0;

#if !defined(HDF5_THREAD_SAFE)
    {
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if (((ierr = nc_inq_varid(fh.get(), var_name.c_str(), &var_id)) != NC_NOERR)
        || ((ierr = nc_inq_var(fh.get(), var_id, nullptr,
        &var_nc_type, &n_dims, dim_id, &n_atts)) != NC_NOERR))
    {
        TECA_ERROR("Failed to query \"" << var_name << "\" variable."
            << std::endl << nc_strerror(ierr))
        return -1;
    }
#if !defined(HDF5_THREAD_SAFE)
    }
#endif

    // convert from the netcdf type code
    NC_DISPATCH(var_nc_type,
       var_type = teca_variant_array_code<NC_T>::get();
       )

    // skip scalars
    if (n_dims == 0)
        return 0;

    std::vector<size_t> dims;
    std::vector<std::string> dim_names;
    unsigned int centering = teca_array_attributes::point_centering;
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
            TECA_ERROR("Failed to query " << ii << "th dimension of variable \""
                << var_name << "\"." << std::endl << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        dim_names.push_back(dim_name);
        dims.push_back(dim);
    }

    atts.set("cf_id", var_id);
    atts.set("cf_dims", dims);
    atts.set("cf_dim_names", dim_names);
    atts.set("cf_type_code", var_nc_type);
    atts.set("type_code", var_type);
    atts.set("centering", centering);

    for (int ii = 0; ii < n_atts; ++ii)
    {
        if (teca_netcdf_util::read_attribute(fh, var_id, ii, atts))
        {
            TECA_ERROR("Failed to read the " << ii << "th attribute for variable \""
                << var_name << "\"." << std::endl << nc_strerror(ierr))
            return -1;
        }
    }

    return 0;
}

// **************************************************************************
int read_variable_attributes(netcdf_handle &fh, int var_id,
    std::string &name, teca_metadata &atts)
{
    int ierr = 0;
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
    if ((ierr = nc_inq_var(fh.get(), var_id, var_name,
            &var_nc_type, &n_dims, dim_id, &n_atts)) != NC_NOERR)
    {
        TECA_ERROR("Failed to query " << var_id << "th variable."
            << std::endl << nc_strerror(ierr))
        return -1;
    }
#if !defined(HDF5_THREAD_SAFE)
    }
#endif

    // convert from the netcdf type code
    NC_DISPATCH(var_nc_type,
       var_type = teca_variant_array_code<NC_T>::get();
       )

    // skip scalars
    if (n_dims == 0)
        return 0;

    std::vector<size_t> dims;
    std::vector<std::string> dim_names;
    unsigned int centering = teca_array_attributes::point_centering;
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
            TECA_ERROR("Failed to query " << ii << "th dimension of variable, "
                << var_name << ". " << std::endl << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        dim_names.push_back(dim_name);
        dims.push_back(dim);
    }

    name = var_name;

    atts.set("cf_id", var_id);
    atts.set("cf_dims", dims);
    atts.set("cf_dim_names", dim_names);
    atts.set("cf_type_code", var_nc_type);
    atts.set("type_code", var_type);
    atts.set("centering", centering);

    for (int ii = 0; ii < n_atts; ++ii)
    {
        if (teca_netcdf_util::read_attribute(fh, var_id, ii, atts))
        {
            TECA_ERROR("Failed to read the " << ii << "th attribute for variable "
                << var_name << ". " << std::endl << nc_strerror(ierr))
            return -1;
        }
    }

    return 0;
}

}
