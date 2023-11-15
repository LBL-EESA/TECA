#include "teca_netcdf_util.h"
#include "teca_common.h"
#include "teca_system_interface.h"
#include "teca_file_util.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"

#include <cstring>
#include <vector>

using namespace teca_variant_array_util;

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
            NC_NT *tmp = static_cast<NC_NT*>(malloc(sizeof(NC_NT)*att_len));
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
int read_variable_attributes(netcdf_handle &fh,
    const std::string &var_name, teca_metadata &atts)
{
    return read_variable_attributes(fh, var_name, "", "", "", "", false, atts);
}

// **************************************************************************
int read_variable_attributes(netcdf_handle &fh, const std::string &var_name,
    const std::string &x_axis_variable, const std::string &y_axis_variable,
    const std::string &z_axis_variable, const std::string &t_axis_variable,
    int clamp_dimensions_of_one, teca_metadata &atts)
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

    // skip scalars
    if (n_dims == 0)
        return 0;

    // convert from the netcdf type code
    NC_DISPATCH(var_nc_type,
       var_type = teca_variant_array_code<NC_NT>::get();
       )

    // read attributes
    for (int ii = 0; ii < n_atts; ++ii)
    {
        if (teca_netcdf_util::read_attribute(fh, var_id, ii, atts))
        {
            TECA_ERROR("Failed to read the " << ii << "th attribute for variable \""
                << var_name << "\"." << std::endl << nc_strerror(ierr))
            return -1;
        }
    }

    // read the dimensions
    int n_mesh_dims = 0;
    int have_mesh_dim[4] = {0};

    int mesh_dim_active[4] = {0};
    int n_active_dims = 0;

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
            TECA_ERROR("Failed to query " << ii << "th dimension of variable \""
                << var_name << "\"." << std::endl << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif

        int active = (clamp_dimensions_of_one && (dim == 1) ? 0 : 1);

        if (!x_axis_variable.empty() &&
            !strcmp(dim_name, x_axis_variable.c_str()))
        {
            have_mesh_dim[0] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[0] = active;
            n_active_dims += active;
        }
        else if (!y_axis_variable.empty() &&
            !strcmp(dim_name, y_axis_variable.c_str()))
        {
            have_mesh_dim[1] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[1] = active;
            n_active_dims += active;
        }
        else if (!z_axis_variable.empty() &&
            !strcmp(dim_name, z_axis_variable.c_str()))
        {
            have_mesh_dim[2] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[2] = active;
            n_active_dims += active;
        }
        else if (!t_axis_variable.empty() &&
            !strcmp(dim_name, t_axis_variable.c_str()))
        {
            have_mesh_dim[3] = 1;
            mesh_dim_active[3] = 1;
        }

        dim_names.push_back(dim_name);
        dims.push_back(dim);
    }

    // can only be point centered if all the dimensions are active coordinate
    // axes
    unsigned int centering = teca_array_attributes::no_centering;
    if (n_mesh_dims == n_dims)
    {
        centering = teca_array_attributes::point_centering;
    }

    atts.set("cf_id", var_id);
    atts.set("cf_dims", dims);
    atts.set("cf_dim_names", dim_names);
    atts.set("cf_type_code", var_nc_type);
    atts.set("type_code", var_type);
    atts.set("centering", centering);
    atts.set("have_mesh_dim", have_mesh_dim, 4);
    atts.set("mesh_dim_active", mesh_dim_active, 4);
    atts.set("n_mesh_dims", n_mesh_dims);
    atts.set("n_active_dims", n_active_dims);

    return 0;
}

// **************************************************************************
int read_variable_attributes(netcdf_handle &fh, int var_id,
    std::string &name, teca_metadata &atts)
{
    return teca_netcdf_util::read_variable_attributes(fh,
        var_id, "", "", "", "", 0, name, atts);
}

// **************************************************************************
int read_variable_attributes(netcdf_handle &fh, int var_id,
    const std::string &x_axis_variable, const std::string &y_axis_variable,
    const std::string &z_axis_variable, const std::string &t_axis_variable,
    int clamp_dimensions_of_one, std::string &name, teca_metadata &atts)
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

    // skip scalars
    if (n_dims == 0)
        return 0;

    name = var_name;

    // convert from the netcdf type code
    NC_DISPATCH(var_nc_type,
       var_type = teca_variant_array_code<NC_NT>::get();
       )

    // read attributes
    for (int ii = 0; ii < n_atts; ++ii)
    {
        if (teca_netcdf_util::read_attribute(fh, var_id, ii, atts))
        {
            TECA_ERROR("Failed to read the " << ii << "th attribute for variable "
                << var_name << ". " << std::endl << nc_strerror(ierr))
            return -1;
        }
    }

    // read the dimensions
    int n_mesh_dims = 0;
    int have_mesh_dim[4] = {0};

    int n_active_dims = 0;
    int mesh_dim_active[4] = {0};

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
            TECA_ERROR("Failed to query " << ii << "th dimension of variable, "
                << var_name << ". " << std::endl << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        int active = (clamp_dimensions_of_one && (dim == 1) ? 0 : 1);

        if (!x_axis_variable.empty() &&
            !strcmp(dim_name, x_axis_variable.c_str()))
        {
            have_mesh_dim[0] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[0] = active;
            n_active_dims += active;
        }
        else if (!y_axis_variable.empty() &&
            !strcmp(dim_name, y_axis_variable.c_str()))
        {
            have_mesh_dim[1] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[1] = active;
            n_active_dims += active;
        }
        else if (!z_axis_variable.empty() &&
            !strcmp(dim_name, z_axis_variable.c_str()))
        {
            have_mesh_dim[2] = 1;
            n_mesh_dims += 1;

            mesh_dim_active[2] = active;
            n_active_dims += active;
        }
        else if (!t_axis_variable.empty() &&
            !strcmp(dim_name, t_axis_variable.c_str()))
        {
            have_mesh_dim[3] = 1;
            mesh_dim_active[3] = 1;
        }

        dim_names.push_back(dim_name);
        dims.push_back(dim);
    }

    // can only be point centered if all the dimensions are active coordinate
    // axes
    unsigned int centering = teca_array_attributes::no_centering;
    if ((n_mesh_dims + have_mesh_dim[3]) == n_dims)
    {
        centering = teca_array_attributes::point_centering;
    }

    atts.set("cf_id", var_id);
    atts.set("cf_dims", dims);
    atts.set("cf_dim_names", dim_names);
    atts.set("cf_type_code", var_nc_type);
    atts.set("type_code", var_type);
    atts.set("centering", centering);
    atts.set("have_mesh_dim", have_mesh_dim);
    atts.set("mesh_dim_active", mesh_dim_active);
    atts.set("n_mesh_dims", n_mesh_dims);
    atts.set("n_active_dims", n_active_dims);

    return 0;
}

// --------------------------------------------------------------------------
read_variable_and_attributes::data_t
read_variable_and_attributes::operator()(int device_id)
{
    (void) device_id;

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
        return this->package(m_id);
    }

    // query variable attributes
    int ierr = 0;
    teca_metadata atts;
    if (teca_netcdf_util::read_variable_attributes(fh,
        m_variable, "", "", "", "", false, atts))
    {
        TECA_ERROR("Failed to read \"" << m_variable << "\" attributes")
        return this->package(m_id);
    }

    // get the type and dimensions
    int var_type = 0;
    int var_id = 0;
    p_teca_size_t_array dims;
    if (atts.get("cf_type_code", var_type)
        || atts.get("cf_id", var_id)
        || !(dims = std::dynamic_pointer_cast<teca_size_t_array>(atts.get("cf_dims"))))
    {
        TECA_ERROR("Metadata issue can't read \"" << m_variable << "\"")
        return this->package(m_id);
    }

    // assume data is on the CPU
    assert(dims->host_accessible());

    // get the size
    size_t var_size = 1;
    size_t start[NC_MAX_DIMS] = {0};
    size_t count[NC_MAX_DIMS] = {0};
    int n_dims = dims->size();
    size_t *p_dims = dims->data();
    for (int i = 0; i < n_dims; ++i)
    {
        var_size *= p_dims[i];
        count[i] = p_dims[i];
    }

    // allocate a buffer and read the variable.
    NC_DISPATCH(var_type,

        auto [var, pvar] = ::New<NC_TT>(var_size);

#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_get_vara(fh.get(), var_id, start, count, pvar)) != NC_NOERR)
        {
            TECA_ERROR("Failed to read variable \"" << m_variable  << "\" from \""
                << m_file << "\". " << nc_strerror(ierr))
            return this->package(m_id);
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif

        // success!
        return this->package(m_id, var, atts);
        )

    // unsupported type
    TECA_ERROR("Failed to read variable \"" << m_variable
        << "\". Unsupported data type")

    return this->package(m_id);
}

// --------------------------------------------------------------------------
read_variable::data_t read_variable::operator()(int device_id)
{
    (void) device_id;

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
        return this->package(m_id);
    }

    // query variable attributes
    int file_id = fh.get();
    int dim_id = 0;
    int var_id = 0;
    int var_ndims = 0;
    size_t var_size = 0;
    nc_type var_type = 0;

    int ierr = 0;
#if !defined(HDF5_THREAD_SAFE)
    {
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    if (((ierr = nc_inq_varid(file_id, m_variable.c_str(), &var_id)) != NC_NOERR)
        || ((ierr = nc_inq_varndims(file_id, var_id, &var_ndims)) != NC_NOERR)
        || (var_ndims != 1)
        || ((ierr = nc_inq_vardimid(file_id, var_id, &dim_id)) != NC_NOERR)
        || ((ierr = nc_inq_dimlen(file_id, dim_id, &var_size)) != NC_NOERR)
        || ((ierr = nc_inq_vartype(file_id, var_id, &var_type)) != NC_NOERR))
    {
        TECA_ERROR("Failed to read metadata for variable \"" << m_variable
            << "\" from \"" << m_file << "\". " << nc_strerror(ierr))
        return this->package(m_id);
    }
#if !defined(HDF5_THREAD_SAFE)
    }
#endif

    // allocate a buffer and read the variable.
    NC_DISPATCH(var_type,
        size_t start = 0;
        auto [var, pvar] = ::New<NC_TT>(var_size);
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_get_vara(file_id, var_id, &start, &var_size, pvar)) != NC_NOERR)
        {
            TECA_ERROR("Failed to read variable \"" << m_variable  << "\" from \""
                << m_file << "\". " << nc_strerror(ierr))
            return this->package(m_id);
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
        // success!
        return this->package(m_id, var);
        )

    // unsupported type
    TECA_ERROR("Failed to read variable \"" << m_variable
        << "\" from \"" << m_file << "\". Unsupported data type")
    return this->package(m_id);
}

// **************************************************************************
int write_variable_attributes(netcdf_handle &fh, int var_id,
    teca_metadata &array_atts)
{
    int ierr = 0;
    unsigned long n_atts = array_atts.size();
    for (unsigned long j = 0; j < n_atts; ++j)
    {
        std::string att_name;
        if (array_atts.get_name(j, att_name))
        {
            TECA_ERROR("failed to get name of the " << j << "th attribute")
            return -1;
        }

        // skip non-standard internal book keeping metadata this is
        // potentially OK to pass through but likely of no interest to
        // anyone else
        if ((att_name == "cf_id") || (att_name == "cf_dims") ||
            (att_name == "cf_dim_names") || (att_name == "type_code") ||
            (att_name == "cf_type_code") || (att_name == "centering") ||
            (att_name == "size") || (att_name == "have_mesh_dim") ||
            (att_name == "mesh_dim_active") || (att_name == "n_mesh_dims") ||
            (att_name == "n_active_dims"))
            continue;

        // get the attribute value
        const_p_teca_variant_array att_values = array_atts.get(att_name);

        // assume the data is on the CPU
        assert(att_values->host_accessible());

        // handle string type
        VARIANT_ARRAY_DISPATCH_CASE(std::string, att_values.get(),
            if (att_values->size() > 1)
                continue;
            auto [patt] = data<CTT>(att_values);
            std::string att_val(patt[0]);
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_put_att_text(fh.get(),
                var_id, att_name.c_str(), att_val.size()+1,
                att_val.c_str())) != NC_NOERR)
            {
                TECA_ERROR("failed to put attribute \"" << att_name << "\"")
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )
        // handle POD types
        else VARIANT_ARRAY_DISPATCH(att_values.get(),

            size_t n_vals = att_values->size();
            int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
            auto [pvals] = data<CTT>(att_values);

#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_put_att(fh.get(), var_id, att_name.c_str(), type,
                n_vals, pvals)) != NC_NOERR)
            {
                TECA_ERROR("failed to put attribute \"" << att_name << "\" "
                    << nc_strerror(ierr))
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )
    }

    return 0;
}

}
