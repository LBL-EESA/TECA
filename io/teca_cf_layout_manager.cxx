#include "teca_cf_layout_manager.h"

#include "teca_config.h"
#include "teca_common.h"
#include "teca_netcdf_util.h"
#include "teca_file_util.h"
#include "teca_array_attributes.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>

// --------------------------------------------------------------------------
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
        this->t = teca_double_array::New(n_indices);

        t->get(this->first_index, this->first_index + n_indices - 1,
            this->t->get());
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_layout_manager::define(const teca_metadata &md_in,
    unsigned long *extent, const std::vector<std::string> &point_arrays,
    const std::vector<std::string> &info_arrays, int compression_level)
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
    if (extent[2] > extent[3])
    {
        TECA_ERROR("Bad y-axis extent [" << extent[2] << ", " << extent[3] << "]")
        return -1;
    }

    // the cf reader always creates 4D data, but some other tools choke
    // on it, notably ParView. All dimensions of 1 are safe to skip, unless
    // we are writing a variable with 1 value.
    size_t nx = extent[1] - extent[0] + 1;
    size_t ny = extent[3] - extent[2] + 1;
    size_t nz = extent[5] - extent[4] + 1;

    unsigned long skip_dim_of_1 = (x && nx > 1 ? 1 : 0) +
        (y && ny > 1 ? 1 : 0) + (z && nz > 1 ? 1 : 0);

    if (this->t)
    {
        coord_arrays[this->n_dims] = this->t;
        coord_array_names[this->n_dims] = this->t_variable.empty() ? "time" : this->t_variable;
        this->dims[this->n_dims] = use_unlimited_dim ? NC_UNLIMITED : this->t->size();
        starts[this->n_dims] = 0;

        if (this->dims[this->n_dims] == NC_UNLIMITED)
            unlimited_dim_actual_size = this->t->size();

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

            starts[this->n_dims] = extent[4];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = nz;

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

            starts[this->n_dims] = extent[2];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = ny;

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

            starts[this->n_dims] = extent[0];

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = nx;

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
        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),
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
        this->var_def[coord_array_names[i]] = std::make_pair(var_id, var_type_code);
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

        int var_id = -1;
        CODE_DISPATCH(var_type_code,
            // define variable
            int var_nc_type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_var(this->handle.get(), name.c_str(), var_nc_type,
                this->n_dims, dim_ids, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("failed to define variable for point array \""
                    << name << "\". " << nc_strerror(ierr))
                return -1;
            }
#if !defined(HDF5_THREAD_SAFE)
            }
#endif
            // save the var id
            this->var_def[name] = std::make_pair(var_id, var_type_code);
            )

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
        if (is_init && ((ierr = nc_var_par_access(this->handle.get(),
            var_id, NC_INDEPENDENT)) != NC_NOERR))
        {
            TECA_ERROR("Failed to set inidependant mode on variable \"" << name << "\"")
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
            this->var_def[name] = std::make_pair(var_id, type_code);
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

        int var_id = it->second.first;
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
        int var_id = it->second.first;

        size_t start = 0;
        // only rank 0 should write these since they are the same on all ranks
        // set the count to be the dimension size (this needs to be an actual
        // size, not NC_UNLIMITED, which results in coordinate arrays not being
        // written)
        size_t count = rank == 0 ? (this->dims[i] == NC_UNLIMITED ?
            unlimited_dim_actual_size : this->dims[i]) : 0;

        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),

            const NT *pa = static_cast<TT*>(coord_arrays[i].get())->get() + starts[i];

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
        size_t counts[4] = {1, 0, 0, 0};

        // make space for the time dimension
        int i0 = this->t ? 1 : 0;

        for (int i = i0; i < this->n_dims; ++i)
            counts[i] = this->dims[i];

        // get this data's position in the file
        unsigned long id = index - this->first_index;
        starts[0] = id;

        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string array_name = point_arrays->get_name(i);
            const_p_teca_variant_array array = point_arrays->get(i);

            // look up the var id
            std::map<std::string, std::pair<int,unsigned int>>::iterator it = this->var_def.find(array_name);
            if (it == this->var_def.end())
            {
                // skip arrays we don't know about. if we want to make this an error
                // we will need the cf writer to subset the point arrays collection
                continue;
                //TECA_ERROR("No var id for \"" << array_name << "\"")
                //return -1;
            }
            int var_id = it->second.first;
            unsigned int declared_type_code = it->second.second;

            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                array.get(),

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of point centered array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                const NT *pa = static_cast<TT*>(array.get())->get();
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
            int var_id = it->second.first;
            unsigned int declared_type_code = it->second.second;

            size_t counts[2] = {1, array->size()};

            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                array.get(),

                unsigned int actual_type_code = teca_variant_array_code<NT>::get();
                if (actual_type_code != declared_type_code)
                {
                    TECA_ERROR("The declared type (" << declared_type_code
                        << ") of information array \"" << array_name
                        << "\" does not match the actual type ("
                        << actual_type_code << ")")
                    return -1;
                }

                const NT *pa = static_cast<TT*>(array.get())->get();
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
