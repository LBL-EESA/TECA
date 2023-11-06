#include "teca_cf_layout_manager.h"

#include "teca_config.h"
#include "teca_common.h"
#include "teca_netcdf_util.h"
#include "teca_file_util.h"
#include "teca_array_attributes.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>

using namespace teca_variant_array_util;

// -------------------------------------/-------------------------------------
int teca_cf_layout_manager::create(const std::string &file_name,
    const std::string &date_format, const teca_metadata &md_in,
    int mode_flags, int use_unlimited_dim)
{
    if (this->file_id < 0 || this->first_index < 0 || this->n_indices < 0)
    {
        TECA_ERROR("Object is improperly intialized")
        return -1;
    }

    // initialize internals
    this->file_name = file_name;
    this->mode_flags = mode_flags;
    this->use_unlimited_dim = use_unlimited_dim;
    this->n_dims = 0;
    for (int i = 0; i < 4; ++i)
        this->dims[i] = 0;

    // get the time axis
    teca_metadata coords;
    if (md_in.get("coordinates", coords))
    {
        TECA_ERROR("failed to get coordinate metadata")
        return -1;
    }

    coords.get("t_variable", this->t_variable);

    p_teca_variant_array t;
    if (!this->t_variable.empty())
        t = coords.get("t");

    // construct the file name
    if (!date_format.empty())
    {
        // get the calendaring metadata
        bool have_calendar = false;
        teca_metadata atts;
        teca_metadata t_atts;
        std::string calendar;
        std::string units;
        if (t && !this->t_variable.empty() &&
            !md_in.get("attributes", atts) &&
            !atts.get(this->t_variable, t_atts) &&
            !t_atts.get("calendar", calendar) &&
            !t_atts.get("units", units))
            have_calendar = true;

        // get the time at the first step written to the file
        double t0 = -1.0;
        if (t)
            t->get(first_index, t0);

        // use the date string for the time information in the filename
        if (!have_calendar)
        {
            // no calendar metadata, fallback to file id
            TECA_WARNING("Metadata is missing time axis and or calendaring "
                "info. The file id will be used in file name instead.")
            teca_file_util::replace_timestep(this->file_name, file_id);
        }
        else if (teca_file_util::replace_time(this->file_name, t0,
            calendar, units, date_format))
        {
            // conversion failed, fall back to file id
            TECA_WARNING("failed to convert relative time value \"" << t0
                << "\" to with the calendar \"" << calendar << "\" units \""
                << units << "\" and format \"" << date_format << "\".")
            teca_file_util::replace_timestep(this->file_name, file_id);
        }
    }
    else
    {
        // use the file id in the filename
        teca_file_util::replace_timestep(this->file_name, file_id);
    }

    // replace extension
    teca_file_util::replace_extension(this->file_name, "nc");

    // create the output file
    if (this->handle.create(this->comm, this->file_name.c_str(), mode_flags))
    {
        TECA_ERROR("failed to create file \"" << file_name << "\"")
        return -1;
    }

    // re-construct the time axis
    if (t)
    {
        this->t = t->new_copy(this->first_index, n_indices);
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::define(const teca_metadata &md_in,
    unsigned long *wextent, const std::vector<std::string> &point_arrays,
    const std::vector<std::string> &info_arrays,
    int collective_buffer, int compression_level)
{
    if (this->defined())
        return 0;

    if (!this->opened())
    {
        TECA_ERROR("Define failed. invalid file handle")
        return -1;
    }

    // rank in the per-file communicator.
    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->comm, &rank);
        MPI_Comm_size(this->comm, &n_ranks);
    }
#endif

    // get the cordinate axes
    teca_metadata coords;
    if (md_in.get("coordinates", coords))
    {
        TECA_ERROR("failed to get coordinate metadata")
        return -1;
    }

    std::string x_variable;
    std::string y_variable;
    std::string z_variable;

    coords.get("x_variable", x_variable);
    coords.get("y_variable", y_variable);
    coords.get("z_variable", z_variable);

    const_p_teca_variant_array x = coords.get("x");
    const_p_teca_variant_array y = coords.get("y");
    const_p_teca_variant_array z = coords.get("z");

    // get the attributes, these are needed to determine the
    // types of the arrays that will be written
    teca_metadata array_attributes;
    if (md_in.get("attributes", array_attributes))
    {
        TECA_ERROR("Array attribues missing from metadata")
        return -1;
    }

    int ierr = NC_NOERR;

    // files are always written in 4D. at least one of the coordinates must
    // have data for the others construct a length one array containing zeros
    const_p_teca_variant_array coord_array
        = x ? x : y ? y : z ? z : this->t ? this->t : nullptr;

    if (!coord_array)
    {
        TECA_ERROR("invalid coordinates")
        return -1;
    }

    size_t starts[4] = {0};
    const_p_teca_variant_array coord_arrays[4];
    std::string coord_array_names[4];
    size_t unlimited_dim_actual_size = 0;

    this->n_dims = 0;
    for (int i = 0; i < 4; ++i)
        this->dims[i] = 0;

    // check for bad bounds request on dataset with y axis in descending order.
    if (wextent[2] > wextent[3])
    {
        TECA_ERROR("Bad y-axis wextent [" << wextent[2] << ", " << wextent[3] << "]")
        return -1;
    }

    // cache the whole_extent to offset the incoming data if this was a subset
    memcpy(this->whole_extent, wextent, 6*sizeof(unsigned long));

    // the cf reader always creates 4D data, but some other tools choke
    // on it, notably ParView. All dimensions of 1 are safe to skip, unless
    // we are writing a variable with 1 value.
    size_t nx = wextent[1] - wextent[0] + 1;
    size_t ny = wextent[3] - wextent[2] + 1;
    size_t nz = wextent[5] - wextent[4] + 1;

    unsigned long skip_dim_of_1 = (x && nx > 1 ? 1 : 0) +
        (y && ny > 1 ? 1 : 0) + (z && nz > 1 ? 1 : 0);

    // record the mesh axis id 0:x, 1:y, 2:z, 3:t
    for (int i = 0; i < 4; ++i)
        this->mesh_axis[i] = 0;

    if (this->t)
    {
        coord_arrays[this->n_dims] = this->t;
        coord_array_names[this->n_dims] = this->t_variable.empty() ? "time" : this->t_variable;
        this->dims[this->n_dims] = use_unlimited_dim ? NC_UNLIMITED : this->t->size();
        starts[this->n_dims] = 0;

        if (this->dims[this->n_dims] == NC_UNLIMITED)
            unlimited_dim_actual_size = this->t->size();

        this->mesh_axis[this->n_dims] = 3;

        ++this->n_dims;
    }
    if (z)
    {
        if (!skip_dim_of_1 || nz > 1)
        {

            coord_arrays[this->n_dims] = z;
            coord_array_names[this->n_dims] = z_variable.empty() ? "z" : z_variable;

            this->dims[this->n_dims] = this->n_dims == 0 &&
                use_unlimited_dim ? NC_UNLIMITED : nz;

            starts[this->n_dims] = wextent[4];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = nz;

            this->mesh_axis[this->n_dims] = 2;

            ++this->n_dims;
        }
    }
    if (y)
    {
        if (!skip_dim_of_1 || ny > 1)
        {
            coord_arrays[this->n_dims] = y;
            coord_array_names[this->n_dims] = y_variable.empty() ? "y" : y_variable;

            this->dims[this->n_dims] = this->n_dims == 0 &&
                use_unlimited_dim ? NC_UNLIMITED : ny;

            starts[this->n_dims] = wextent[2];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = ny;

            this->mesh_axis[this->n_dims] = 1;

            ++this->n_dims;
        }
    }
    if (x)
    {
        if (!skip_dim_of_1 || nx > 1)
        {
            coord_arrays[this->n_dims] = x;
            coord_array_names[this->n_dims] = x_variable.empty() ? "x" : x_variable;

            this->dims[this->n_dims] = this->n_dims == 0 &&
                 use_unlimited_dim ? NC_UNLIMITED : nx;

            starts[this->n_dims] = wextent[0];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = nx;

            this->mesh_axis[this->n_dims] = 0;

            ++this->n_dims;
        }
    }

    // build the dictionary of names to ncids
    int dim_ids[4] = {-1};
    for (int i = 0; i < this->n_dims; ++i)
    {
        // define dimension
        int dim_id = -1;
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_def_dim(this->handle.get(), coord_array_names[i].c_str(),
            this->dims[i], &dim_id)) != NC_NOERR)
        {
            TECA_ERROR("failed to define dimensions for coordinate axis "
                <<  i << " \"" << coord_array_names[i] << "\" "
                << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif

        // save the dim id
        dim_ids[i] = dim_id;

        // define variable for the axis
        int var_id = -1;
        unsigned int var_type_code = 0;
        VARIANT_ARRAY_DISPATCH(coord_arrays[i].get(),
            var_type_code = teca_variant_array_code<NT>::get();
            int var_nc_type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_var(this->handle.get(), coord_array_names[i].c_str(),
                var_nc_type, 1, &dim_id, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("failed to define variables for coordinate axis "
                    <<  i << " \"" << coord_array_names[i] << "\" "
                    << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )

        // save the var id
        this->var_def[coord_array_names[i]] = var_def_t(var_id, var_type_code);
    }

    // define variables for each point array
    unsigned int n_arrays = point_arrays.size();
    for (unsigned int i = 0; i < n_arrays; ++i)
    {
        std::string name = point_arrays[i];

        teca_metadata atts_i;
        if (array_attributes.get(name, atts_i))
        {
            TECA_ERROR("failed to get attributes for point array \""
                << name << "\"")
            return -1;
        }

        // get the type code
        int var_type_code = 0;
        if (atts_i.get("type_code", var_type_code))
        {
            TECA_ERROR("Array \"" << name << "\" missing type attribute")
            return -1;
        }

        // get the active dimensions
        int dim_active[4] = {0};
        if (atts_i.get("mesh_dim_active", dim_active, 4))
        {
            // this matches the original behavior, but now that 2D and 3D data
            // is supported in the same dataset all variables should provide
            // the active dims metadata
            if (rank == 0)
                TECA_WARNING("attributes for \"" << name << "\" are missing the"
                    " mesh_dim_active key. All mesh dimensions are assumed to be active.")

            for (int j = 0; j < 4; ++j)
                dim_active[j] = 1;
        }

        // copy the dim ids from the active dimensions
        std::array<int,4> adims{{0,0,0,0}};
        int n_active = 0;
        int active_dim_ids[4] = {-1};
        for (int j = 0; j < this->n_dims; ++j)
        {
            int active = dim_active[this->mesh_axis[j]];
            adims[j] = active;
            if (active)
            {
                active_dim_ids[n_active] = dim_ids[j];
                ++n_active;
            }
        }

        int var_id = -1;
        CODE_DISPATCH(var_type_code,
            // define variable
            int var_nc_type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_var(this->handle.get(), name.c_str(), var_nc_type,
                n_active, active_dim_ids, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("failed to define variable for point array \""
                    << name << "\". " << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )

       // save the variable definition
       this->var_def[name] = var_def_t(var_id, var_type_code, adims);

#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        // turn on compression for point arrays
        if ((compression_level > 0) &&
            ((ierr = nc_def_var_deflate(this->handle.get(),
                var_id, 0, 1, compression_level) != NC_NOERR)))
        {
            TECA_ERROR("failed to set compression level to "
                << compression_level << " for point array \""
                << name << "\". " << nc_strerror(ierr))
            return -1;
        }

#if defined(TECA_HAS_NETCDF_MPI)
        int access_mode = collective_buffer ? NC_COLLECTIVE : NC_INDEPENDENT;
        if (is_init && ((ierr = nc_var_par_access(this->handle.get(),
            var_id, access_mode)) != NC_NOERR))
        {
            TECA_ERROR("Failed to set "
                << (collective_buffer ? "collective" : "independant")
                << " access  mode on variable \"" << name << "\"")
            return -1;
        }
#endif
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
    }

    // add dimension and define information arrays
    int n_info_dims = this->t ? 2 : 1;
    n_arrays = info_arrays.size();
    for (unsigned int i = 0; i < n_arrays; ++i)
    {
        std::string name = info_arrays[i];

        // get the attributes. Need type and size for the definition
        teca_metadata array_attributes_i;
        if (array_attributes.get(name, array_attributes_i))
        {
            TECA_ERROR("failed to get attributes for info array \""
                << name << "\"")
            return -1;
        }

        size_t size = 0;
        unsigned int type_code = 0;
        if ( array_attributes_i.get("type_code", type_code) ||
            array_attributes_i.get("size", size) || (type_code < 1) || (size < 1))
        {
            TECA_ERROR("invalid attributes for info array \"" << name
                << "\" missing or invlaid type and/or size")
            return -1;
        }

        // define dimension
        std::string dim_name = "dim_" + name;
        int dim_id = -1;
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if ((ierr = nc_def_dim(this->handle.get(), dim_name.c_str(),
            size, &dim_id)) != NC_NOERR)
        {
            TECA_ERROR("failed to define dimensions for information array "
                <<  i << " \"" << name << "\" " << nc_strerror(ierr))
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif

        // set up dim ids for definition
        int info_dim_ids[2];
        if (this->t)
        {
            info_dim_ids[0] = dim_ids[0];
            info_dim_ids[1] = dim_id;
        }
        else
        {
            info_dim_ids[0] = dim_id;
            info_dim_ids[1] = 0;
        }

        int var_id = -1;
        CODE_DISPATCH(type_code,
            // define variable
            int var_nc_type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_var(this->handle.get(), name.c_str(), var_nc_type,
                n_info_dims, info_dim_ids, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("failed to define variable for information array \""
                    << name << "\". " << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            // save the var id
            this->var_def[name] = var_def_t(var_id, type_code);
            )

#if defined(TECA_HAS_NETCDF_MPI)
#if !defined(HDF5_THREAD_SAFE)
        {
        std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
        if (is_init && ((ierr = nc_var_par_access(this->handle.get(), var_id,
            NC_INDEPENDENT)) != NC_NOERR))
        {
            TECA_ERROR("Failed to set independent mode on variable \"" << name << "\"")
            return -1;
        }
#if !defined(HDF5_THREAD_SAFE)
        }
#endif
#endif
    }

    // write attributes of the varibles in hand
    std::map<std::string, var_def_t>::iterator it = this->var_def.begin();
    std::map<std::string, var_def_t>::iterator end = this->var_def.end();
    for (; it != end; ++it)
    {
        teca_metadata array_atts;
        std::string array_name = it->first;
        if (array_attributes.get(array_name, array_atts))
        {
            // It's ok for a variable not to have attributes
            continue;
        }

        int var_id = it->second.var_id;
        if (teca_netcdf_util::write_variable_attributes(
            this->handle, var_id, array_atts))
        {
            TECA_ERROR("Failed to write attributes for \"" << array_name << "\"")
        }
    }

#if !defined(HDF5_THREADAFE)
    {
    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
    // prevent NetCDF from writing 2x.
    int old_fill = 0;
    if ((ierr = nc_set_fill(this->handle.get(), NC_NOFILL, &old_fill)) != NC_NOERR)
    {
        TECA_ERROR("Failed to disable fill mode on file " << this->file_id)
    }

    // end metadata definition phase
    ierr = nc_enddef(this->handle.get());
#if !defined(HDF5_THREAD_SAFE)
    }
#endif

    // write the coordinate arrays
    for (int i = 0; i < this->n_dims; ++i)
    {
        // look up the var id
        std::string array_name = coord_array_names[i];
        std::map<std::string, var_def_t>::iterator it = this->var_def.find(array_name);
        if (it  == this->var_def.end())
        {
            TECA_ERROR("No var id for \"" << array_name << "\"")
            return -1;
        }
        int var_id = it->second.var_id;

        size_t start = 0;
        // only rank 0 should write these since they are the same on all ranks
        // set the count to be the dimension size (this needs to be an actual
        // size, not NC_UNLIMITED, which results in coordinate arrays not being
        // written)
        size_t count = rank == 0 ? (this->dims[i] == NC_UNLIMITED ?
            unlimited_dim_actual_size : this->dims[i]) : 0;

        VARIANT_ARRAY_DISPATCH(coord_arrays[i].get(),

            auto [spa, pa] = get_host_accessible<CTT>(coord_arrays[i]);

            pa += starts[i];

            sync_host_access_any(coord_arrays[i]);

#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_put_vara(this->handle.get(), var_id, &start, &count, pa)) != NC_NOERR)
            {
                TECA_ERROR("failed to write \"" << coord_array_names[i] << "\" axis. "
                    << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            )
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::write(long index,
    const const_p_teca_array_collection &point_arrays,
    const const_p_teca_array_collection &info_arrays)
{
    if (!this->opened())
    {
        TECA_ERROR("Write failed. invalid file handle")
        return -1;
    }

    if (!this->defined())
    {
        TECA_ERROR("Write failed. file layout has not been defined")
        return -1;
    }

    int ierr = NC_NOERR;

    // write point arrays
    if (unsigned int n_arrays = point_arrays->size())
    {
        size_t starts[4] = {0, 0, 0, 0};

        // get this data's position in the file
        unsigned long id = index - this->first_index;
        starts[0] = id;

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name = point_arrays->get_name(i);
            const_p_teca_variant_array array = point_arrays->get(i);

            // look up the var id
            std::map<std::string, var_def_t>::iterator it = this->var_def.find(array_name);
            if (it == this->var_def.end())
            {
                // skip arrays we don't know about. if we want to make this an error
                // we will need the cf writer to subset the point arrays collection
                continue;
                //TECA_ERROR("No var id for \"" << array_name << "\"")
                //return -1;
            }
            int var_id = it->second.var_id;
            unsigned int declared_type_code = it->second.type_code;
            const std::array<int,4> &active_dims = it->second.active_dims;

            // make space for the time dimension
            size_t counts[4] = {1, 0, 0, 0};
            int j0 = this->t && active_dims[0] ? 1 : 0;
            int n_active = j0;
            for (int j = j0; j < this->n_dims; ++j)
            {
                if (active_dims[j])
                {
                    counts[n_active] = this->dims[j];
                    ++n_active;
                }
            }

            VARIANT_ARRAY_DISPATCH(array.get(),

                auto [spa, pa] = get_host_accessible<CTT>(array);

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of point centered array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                sync_host_access_any(array);

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                {
                    TECA_ERROR("failed to write point array \"" << array_name << "\". "
                        << nc_strerror(ierr))
                    return -1;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                )
        }
    }

    // write the information arrays
    if (unsigned int n_arrays = info_arrays->size())
    {
        size_t starts[2] = {0, 0};

        // get this data's position in the file
        unsigned long id = index - this->first_index;
        starts[0] = id;

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name = info_arrays->get_name(i);
            const_p_teca_variant_array array = info_arrays->get(i);

            // look up the var id
            std::map<std::string, var_def_t>::iterator it = this->var_def.find(array_name);
            if (it  == this->var_def.end())
            {
                // skip arrays we don't know about. if we want to make this an error
                // we will need the cf writer to subset the information arrays collection
                continue;
                //TECA_ERROR("No var id for \"" << array_name << "\"")
                //return -1;
            }
            int var_id = it->second.var_id;
            unsigned int declared_type_code = it->second.type_code;

            size_t counts[2] = {1, array->size()};

            VARIANT_ARRAY_DISPATCH(array.get(),

                auto [spa, pa] = get_host_accessible<CTT>(array);

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of information array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                sync_host_access_any(array);

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                {
                    TECA_ERROR("failed to write information array \"" << array_name << "\". "
                        << nc_strerror(ierr))
                    return -1;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                )
        }
    }

    // keep track of what's been done
    this->n_written += 1;

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::write(const unsigned long extent[6],
    const unsigned long temporal_extent[2],
    const const_p_teca_array_collection &point_arrays,
    const const_p_teca_array_collection &info_arrays)
{
    if (!this->opened())
    {
        TECA_ERROR("Write failed. invalid file handle")
        return -1;
    }

    if (!this->defined())
    {
        TECA_ERROR("Write failed. file layout has not been defined")
        return -1;
    }

    int ierr = NC_NOERR;

    unsigned long temporal_extent_size = temporal_extent[1] - temporal_extent[0] + 1;

    // intersect the teporal extent of the passed arrays and the file
    unsigned long active_extent[2];
    active_extent[0] = std::max<unsigned long>(this->first_index, temporal_extent[0]);
    active_extent[1] = std::min<unsigned long>(this->first_index + this->n_indices - 1, temporal_extent[1]);

    // write point arrays
    if (unsigned int n_arrays = point_arrays->size())
    {
        size_t starts[4] = {0, 0, 0, 0};
        size_t counts[4] = {0, 0, 0, 0};

        // get this data's position in the file
        starts[0] = active_extent[0] - this->first_index;
        counts[0] = active_extent[1] - active_extent[0] + 1;

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name = point_arrays->get_name(i);
            const_p_teca_variant_array array = point_arrays->get(i);

            // look up the var id
            std::map<std::string, var_def_t>::iterator it = this->var_def.find(array_name);
            if (it == this->var_def.end())
            {
                // skip arrays we don't know about. if we want to make this an error
                // we will need the cf writer to subset the point arrays collection
                continue;
                //TECA_ERROR("No var id for \"" << array_name << "\"")
                //return -1;
            }
            int var_id = it->second.var_id;
            unsigned int declared_type_code = it->second.type_code;
            const std::array<int,4> &active_dims = it->second.active_dims;

            // make space for the time dimension
            int j0 = this->t && active_dims[0] ? 1 : 0;
            int n_active = j0;

            // compute the start and count for this chunk of data, taking into
            // account any runtime specified subsetting
            for (int j = j0; j < this->n_dims; ++j)
            {
                if (active_dims[j])
                {
                    int q = 2*this->mesh_axis[j];

                    starts[n_active] = extent[q] - this->whole_extent[q];
                    counts[n_active] = extent[q + 1] - extent[q] + 1;

                    ++n_active;
                }
            }

            VARIANT_ARRAY_DISPATCH(array.get(),

                auto [spa, pa] = get_host_accessible<CTT>(array);

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of point centered array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                sync_host_access_any(array);

                // advance the pointer to the first time step that will be
                // written by this manager
                size_t step_size = array->size() / temporal_extent_size;
                pa += (active_extent[0] - temporal_extent[0]) * step_size;

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                {
                    TECA_ERROR("failed to write point array \"" << array_name << "\". "
                        << nc_strerror(ierr))
                    return -1;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                )
        }
    }

    // write the information arrays
    if (unsigned int n_arrays = info_arrays->size())
    {
        size_t starts[2] = {0, 0};
        size_t counts[2] = {0, 0};

        // get this data's position in the file
        starts[0] = active_extent[0] - this->first_index;
        counts[0] = active_extent[1] - active_extent[0] + 1;

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name = info_arrays->get_name(i);
            const_p_teca_variant_array array = info_arrays->get(i);

            // look up the var id
            std::map<std::string, var_def_t>::iterator it = this->var_def.find(array_name);
            if (it  == this->var_def.end())
            {
                // skip arrays we don't know about. if we want to make this an error
                // we will need the cf writer to subset the information arrays collection
                continue;
                //TECA_ERROR("No var id for \"" << array_name << "\"")
                //return -1;
            }
            int var_id = it->second.var_id;
            unsigned int declared_type_code = it->second.type_code;

            VARIANT_ARRAY_DISPATCH(array.get(),

                auto [spa, pa] = get_host_accessible<CTT>(array);

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of information array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                // advance the pointer to the first time step that will be
                // written by this manager
                size_t step_size = array->size() / temporal_extent_size;
                pa += (active_extent[0] - temporal_extent[0]) * step_size;

                // get the size of the subset of the array actually written
                counts[1] = (active_extent[1] - active_extent[0] + 1) * step_size;

                sync_host_access_any(array);

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                {
                    TECA_ERROR("failed to write information array \"" << array_name << "\". "
                        << nc_strerror(ierr))
                    return -1;
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                )
        }
    }

    // keep track of what's been done
    this->n_written += active_extent[1] - active_extent[0] + 1;

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::flush()
{
    return this->handle.flush();
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::to_stream(std::ostream &os)
{
    int frank = 0;
    int n_franks = 1;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->comm, &frank);
        MPI_Comm_size(this->comm, &n_franks);
    }
#endif
    os << "file_rank=" << frank << " n_file_ranks="
       << n_franks << " file_id=" << file_id
       << " file_name=\"" << this->file_name
       << "\" first_index=" << this->first_index
       << " n_indices=" << this->n_indices
       << " n_written=" << this->n_written;

    return 0;
}
