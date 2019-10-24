#include "teca_cf_writer.h"

#include "teca_config.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_file_util.h"
#include "teca_netcdf_util.h"
#include "teca_coordinate_util.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


class teca_cf_writer::internals_t
{
public:
    struct time_series_layout
    {
        time_series_layout() : file_id(0), first_index(0),
            n_indices(0), n_written(0), n_dims(0),
            dims{0} {}

        // creates the NetCDF file.
        int create(unsigned long file_id,
            unsigned long first_index, unsigned long n_indices,
            const std::string &file_name, const std::string &date_format,
            const teca_metadata &md_in, int mode_flags, int use_unlimited_dim);

        // defines the NetCDF file layout.
        int define(const const_p_teca_variant_array &x,
            const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
            const std::string &x_variable, const std::string &y_variable,
            const std::string &z_variable,
            const std::vector<const_p_teca_array_collection> &point_arrays,
            const std::vector<const_p_teca_array_collection> &info_arrays,
            int compression_level, const teca_metadata &md_in);

        // writes the colllection of arrays to the NetCDF file
        // in the correct spot.
        int write(const std::vector<long> &request_ids,
            const std::vector<const_p_teca_array_collection> &point_arrays,
            const std::vector<const_p_teca_array_collection> &info_arrays);

        int close(){ return this->handle.close(); }

        bool opened() { return bool(this->handle); }
        bool defined() { return this->n_dims > 0; }

        bool completed()
        {
            return this->n_written == this->n_indices;
        }

        // identifying the file
        unsigned long file_id;
        std::string file_name;
        teca_netcdf_util::netcdf_handle handle;

        // for indentifying the incoming dataset and determining its
        // position in the file
        unsigned long first_index;
        unsigned long n_indices;
        unsigned long n_written;

        // for low level NetCDF book keeping
        int mode_flags;
        int use_unlimited_dim;
        int n_dims;
        size_t dims[4];
        std::map<std::string, int> var_ids;
        std::string t_variable;
        p_teca_double_array t;
    };

    // the file table maps from a file_id to a specific layout manager
    using file_table_t = std::unordered_map<unsigned long, time_series_layout>;
    using file_table_iterator_t = file_table_t::iterator;
    file_table_t file_table;
};

// --------------------------------------------------------------------------
int teca_cf_writer::internals_t::time_series_layout::create(
    unsigned long file_id, unsigned long first_index, unsigned long n_indices,
    const std::string &file_name, const std::string &date_format,
    const teca_metadata &md_in, int mode_flags, int use_unlimited_dim)
{
    // initialize internals
    this->file_id = file_id;
    this->first_index = first_index;
    this->n_indices = n_indices;
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
    p_teca_variant_array t = coords.get("t");

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
    if (this->handle.create(this->file_name.c_str(), mode_flags))
    {
        TECA_ERROR("failed to create file \"" << file_name << "\"")
        return -1;
    }

    // re-construct the time axis
    if (t)
    {
        this->t = teca_double_array::New(n_indices);

        t->get(first_index, first_index + n_indices - 1,
            this->t->get());
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_writer::internals_t::time_series_layout::define(
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, const std::string &x_variable,
    const std::string &y_variable, const std::string &z_variable,
    const std::vector<const_p_teca_array_collection> &point_arrays,
    const std::vector<const_p_teca_array_collection> &info_arrays,
    int compression_level, const teca_metadata &md_in)
{
    if (this->defined())
        return 0;

    if (!this->opened())
    {
        TECA_ERROR("Define failed. invalid file handle")
        return -1;
    }

    // get the attributes
    teca_metadata array_attributes;
    if (md_in.get("attributes", array_attributes) && !this->t_variable.empty())
    {
        // array attributes are not necessary to write the data
        // if no attributes then try to pass calendaring information through
        std::string calendar;
        std::string time_units;
        if (!md_in.get("calendar", calendar) &&
            !md_in.get("time_units", time_units))
        {
            teca_metadata t_atts;
            t_atts.set("calendar", calendar);
            t_atts.set("time_units", time_units);
            array_attributes.set(this->t_variable, t_atts);
        }
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

    const_p_teca_variant_array coord_arrays[4];
    std::string coord_array_names[4];
    size_t unlimited_dim_actual_size = 0;

    this->n_dims = 0;
    for (int i = 0; i < 4; ++i)
        this->dims[i] = 0;

    // the cf reader always creates 4D data, but some other tools choke
    // on it, notably ParView. All dimensions of 1 are safe to skip, unless
    // we are writing a variable with 1 value.
    unsigned long skip_dim_of_1 = (x && x->size() > 1 ? 1 : 0) +
        (y && y->size() > 1 ? 1 : 0) + (z && z->size() > 1 ? 1 : 0);

    if (this->t)
    {
        coord_arrays[this->n_dims] = this->t;
        coord_array_names[this->n_dims] = this->t_variable.empty() ? "time" : this->t_variable;
        this->dims[this->n_dims] = use_unlimited_dim ? NC_UNLIMITED : this->t->size();

        if (this->dims[this->n_dims] == NC_UNLIMITED)
            unlimited_dim_actual_size = this->t->size();

        ++this->n_dims;
    }
    if (z)
    {
        if (!skip_dim_of_1 || z->size() > 1)
        {
            coord_arrays[this->n_dims] = z;
            coord_array_names[this->n_dims] = z_variable.empty() ? "z" : z_variable;
            this->dims[this->n_dims] = this->n_dims == 0 &&
                use_unlimited_dim ? NC_UNLIMITED : z->size();

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = z->size();

            ++this->n_dims;
        }
    }
    if (y)
    {
        if (!skip_dim_of_1 || y->size() > 1)
        {
            coord_arrays[this->n_dims] = y;
            coord_array_names[this->n_dims] = y_variable.empty() ? "y" : y_variable;

            this->dims[this->n_dims] = this->n_dims == 0 &&
                use_unlimited_dim ? NC_UNLIMITED : y->size();

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = y->size();

            ++this->n_dims;
        }
    }
    if (x)
    {
        if (!skip_dim_of_1 || x->size() > 1)
        {
            coord_arrays[this->n_dims] = x;
            coord_array_names[this->n_dims] = x_variable.empty() ? "x" : x_variable;

            this->dims[this->n_dims] = this->n_dims == 0 &&
                 use_unlimited_dim ? NC_UNLIMITED : x->size();

            if (this->dims[this->n_dims] == NC_UNLIMITED)
                unlimited_dim_actual_size = x->size();

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
        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),
            int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_var(this->handle.get(), coord_array_names[i].c_str(),
                type, 1, &dim_id, &var_id)) != NC_NOERR)
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
        this->var_ids[coord_array_names[i]] = var_id;
    }

    // define variables for each point array
    if (point_arrays.size())
    {
        unsigned int n_arrays = point_arrays[0]->size();
        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string name = point_arrays[0]->get_name(i);
            const_p_teca_variant_array array = point_arrays[0]->get(i);

            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                array.get(),
                // define variable
                int var_id = -1;
                int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_def_var(this->handle.get(), name.c_str(), type,
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
                this->var_ids[name] = var_id;
                )

            // turn on compression for point arrays
            if ((compression_level > 0) &&
                ((ierr = nc_def_var_deflate(this->handle.get(),
                    this->var_ids[name], 0, 1, compression_level) != NC_NOERR)))
            {
                TECA_ERROR("failed to set compression level to "
                    << compression_level << " for point array \""
                    << name << "\". " << nc_strerror(ierr))
                return -1;
            }
        }
    }

    // add dimension and define information arrays
    int n_info_dims = this->t ? 2 : 1;
    if (info_arrays.size())
    {
        unsigned int n_arrays = info_arrays[0]->size();
        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            const_p_teca_variant_array array = info_arrays[0]->get(i);
            std::string name = info_arrays[0]->get_name(i);

            // define dimension
            std::string dim_name = "dim_" + name;
            int dim_id = -1;
#if !defined(HDF5_THREAD_SAFE)
            {
            std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
            if ((ierr = nc_def_dim(this->handle.get(), dim_name.c_str(),
                array->size(), &dim_id)) != NC_NOERR)
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

            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                array.get(),
                // define variable
                int var_id = -1;
                int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_def_var(this->handle.get(), name.c_str(), type,
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
                this->var_ids[name] = var_id;
                )
        }
    }

    // write attributes of the varibles in hand
    std::map<std::string,int>::iterator it = this->var_ids.begin();
    std::map<std::string,int>::iterator end = this->var_ids.end();
    for (; it != end; ++it)
    {
        teca_metadata array_atts;
        std::string array_name = it->first;
        if (array_attributes.get(array_name, array_atts))
        {
            // It's ok for a variable not to have attributes
            continue;
        }

        int var_id = it->second;

        unsigned long n_atts = array_atts.size();
        for (unsigned long j = 0; j < n_atts; ++j)
        {
            std::string att_name;
            if (array_atts.get_name(j, att_name))
            {
                TECA_ERROR("failed to get name of the " << j
                    << "th attribute for array \"" << array_name << "\"")
                return -1;
            }

            // TODO -- remove these from the metadata perhaps in the reader?
            // skip non-standard internal book keeping metadata this is
            // potentially OK to pass through but likely of no interest to
            // anyone else
            if ((att_name == "id") || (att_name == "dims") ||
                (att_name == "dim_names") || (att_name == "type") ||
                (att_name == "centering"))
                continue;

            // get the attribute value
            const_p_teca_variant_array att_values = array_atts.get(att_name);

            // handle string type
            TEMPLATE_DISPATCH_CLASS(
                const teca_variant_array_impl, std::string,
                att_values.get(),
                if (att_values->size() > 1)
                    continue;
                const std::string att_val = static_cast<const TT*>(att_values.get())->get(0);
#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_att_text(this->handle.get(), var_id, att_name.c_str(), att_val.size()+1,
                    att_val.c_str())) != NC_NOERR)
                {
                    TECA_ERROR("failed to put attribute \"" << att_name << "\"")
                }
#if !defined(HDF5_THREAD_SAFE)
                }
#endif
                continue;
                )

            // handle POD types
            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                att_values.get(),

                int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
                const NT *pvals = static_cast<TT*>(att_values.get())->get();
                unsigned long n_vals = att_values->size();

#if !defined(HDF5_THREAD_SAFE)
                {
                std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                if ((ierr = nc_put_att(this->handle.get(), var_id, att_name.c_str(), type,
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
    }

#if !defined(HDF5_THREAD_SAFE)
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
        std::map<std::string, int>::iterator it = this->var_ids.find(array_name);
        if (it  == this->var_ids.end())
        {
            TECA_ERROR("No var id for \"" << array_name << "\"")
            return -1;
        }
        int var_id = it->second;

        size_t start = 0;
        // set the count to be the dimension size (this needs to be an actual
        // size, not NC_UNLIMITED, which results in coordinate arrays not being
        // written)
        size_t count = this->dims[i] == NC_UNLIMITED ?
            unlimited_dim_actual_size : this->dims[i];

        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),
            const NT *pa = static_cast<TT*>(coord_arrays[i].get())->get();
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
int teca_cf_writer::internals_t::time_series_layout::write(
    const std::vector<long> &request_ids,
    const std::vector<const_p_teca_array_collection> &point_arrays,
    const std::vector<const_p_teca_array_collection> &info_arrays)
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

    // get the number of chuncks of data to write
    unsigned int n_ids = request_ids.size();

    // write point arrays
    if (point_arrays.size())
    {
        size_t starts[4] = {0, 0, 0, 0};
        size_t counts[4] = {1, 0, 0, 0};
        for (int i = 1; i < this->n_dims; ++i)
            counts[i] = this->dims[i];

        for (unsigned int q = 0; q < n_ids; ++q)
        {
            // get this data's position in the file
            unsigned long id = request_ids[q] - this->first_index;
            starts[0] = id;

            unsigned int n_arrays = point_arrays[q]->size();
            for (unsigned int i = 0; i < n_arrays; ++i)
            {
                std::string array_name = point_arrays[q]->get_name(i);
                const_p_teca_variant_array array = point_arrays[q]->get(i);

                // look up the var id
                std::map<std::string, int>::iterator it = this->var_ids.find(array_name);
                if (it  == this->var_ids.end())
                {
                    TECA_ERROR("No var id for \"" << array_name << "\"")
                    return -1;
                }
                int var_id = it->second;

                TEMPLATE_DISPATCH(const teca_variant_array_impl,
                    array.get(),
                    const NT *pa = static_cast<TT*>(array.get())->get();
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                    {
                        TECA_ERROR("failed to write array \"" << array_name << "\". "
                            << nc_strerror(ierr))
                        return -1;
                    }
#if !defined(HDF5_THREAD_SAFE)
                    }
#endif
                    )
            }
        }
    }

    // write the information arrays
    if (info_arrays.size())
    {
        size_t starts[2] = {0, 0};
        for (unsigned int q = 0; q < n_ids; ++q)
        {
            // get this data's position in the file
            unsigned long id = request_ids[q] - this->first_index;
            starts[0] = id;

            unsigned int n_arrays = info_arrays[q]->size();
            for (unsigned int i = 0; i < n_arrays; ++i)
            {
                std::string array_name = info_arrays[q]->get_name(i);
                const_p_teca_variant_array array = info_arrays[q]->get(i);

                size_t counts[2] = {1, array->size()};

                // look up the var id
                std::map<std::string, int>::iterator it = this->var_ids.find(array_name);
                if (it  == this->var_ids.end())
                {
                    TECA_ERROR("No var id for \"" << array_name << "\"")
                    return -1;
                }
                int var_id = it->second;

                TEMPLATE_DISPATCH(const teca_variant_array_impl,
                    array.get(),
                    const NT *pa = static_cast<TT*>(array.get())->get();
#if !defined(HDF5_THREAD_SAFE)
                    {
                    std::lock_guard<std::mutex> lock(teca_netcdf_util::get_netcdf_mutex());
#endif
                    if ((ierr = nc_put_vara(this->handle.get(), var_id, starts, counts, pa)) != NC_NOERR)
                    {
                        TECA_ERROR("failed to write array \"" << array_name << "\". "
                            << nc_strerror(ierr))
                        return -1;
                    }
#if !defined(HDF5_THREAD_SAFE)
                    }
#endif
                    )
            }
        }
    }

    // keep track of what's been done
    this->n_written += request_ids.size();

    return 0;
}

// --------------------------------------------------------------------------
teca_cf_writer::teca_cf_writer() :
    file_name(""), date_format("%F-%HZ"), first_step(0), last_step(-1),
    steps_per_file(1), mode_flags(NC_CLOBBER|NC_NETCDF4), use_unlimited_dim(1),
    compression_level(-1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
    this->set_stream_size(1);
    this->internals = new teca_cf_writer::internals_t;
}

// --------------------------------------------------------------------------
teca_cf_writer::~teca_cf_writer()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cf_writer::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cf_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, file_name,
            "path/name to write series to")
        TECA_POPTS_GET(std::string, prefix, date_format,
            "strftime format string for date string in output filename (%F-%H)")
        TECA_POPTS_GET(long, prefix, first_step,
            "set the first time step to process (0)")
        TECA_POPTS_GET(long, prefix, last_step,
            "set the last time step to process. A value less than 0 results "
            "in all steps being processed.(-1)")
        TECA_POPTS_GET(unsigned int, prefix, steps_per_file,
            "set the number of time steps to write per file (1)")
        TECA_POPTS_GET(int, prefix, mode_flags,
            "mode flags to pass to NetCDF when creating the file (NC_CLOBBER)")
        TECA_POPTS_GET(int, prefix, use_unlimited_dim,
            "if set the slowest varying dimension is specified to be "
            "NC_UNLIMITED. (1)")
        TECA_POPTS_GET(int, prefix, compression_level,
            "sets the zlib compression level used for each variable;"
            "does nothing if the value is less than or equal to 0. (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, std::string, prefix, date_format)
    TECA_POPTS_SET(opts, long, prefix, first_step)
    TECA_POPTS_SET(opts, long, prefix, last_step)
    TECA_POPTS_SET(opts, unsigned int, prefix, steps_per_file)
    TECA_POPTS_SET(opts, int, prefix, mode_flags)
    TECA_POPTS_SET(opts, int, prefix, use_unlimited_dim)
    TECA_POPTS_SET(opts, int, prefix, compression_level)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_cf_writer::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cf_writer::get_output_metadata" << std::endl;
#endif
    (void)port;

#if defined(TECA_HAS_MPI)
    int rank = 0;
    int n_ranks = 1;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
        MPI_Comm_size(this->get_communicator(), &n_ranks);
    }
#endif

    const teca_metadata &md_in = input_md[0];

    // upstream indices span some set of work to parallelize over,
    // the writer converts these into some number of files. here
    // make the conversion to present the data as a number of
    // files to write.
    std::string up_initializer_key;
    if (md_in.get("index_initializer_key", up_initializer_key))
    {
        TECA_ERROR("Failed to locate index_initializer_key")
        return teca_metadata();
    }

    std::string up_request_key;
    if (md_in.get("index_request_key", up_request_key))
    {
        TECA_ERROR("Failed to locate index_request_key")
        return teca_metadata();
    }

    long n_indices = 0;
    if (md_in.get(up_initializer_key, n_indices))
    {
        TECA_ERROR("Missing index initializer \"" << up_initializer_key << "\"")
        return teca_metadata();
    }

    // deal with subsetting of time steps
    long idx_0 = this->first_step < 0 ? 0 : this->first_step;
    long idx_1 = this->last_step < 1 ? n_indices - 1 : this->last_step;
    n_indices = idx_1 - idx_0 + 1;

    // estimate the number of files we create for this run
    long n_steps_left = n_indices % this->steps_per_file;
    long n_files = n_indices / this->steps_per_file + (n_steps_left ? 1 : 0);

    // pass through input metadata
    teca_metadata md_out(md_in);

    // set up executive control keys
    md_out.remove(up_initializer_key);
    md_out.set("index_initializer_key", std::string("number_of_files"));
    md_out.set("index_request_key", std::string("file_id"));
    md_out.set("number_of_files", n_files);

    return md_out;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cf_writer::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cf_writer::get_upstream_request" << std::endl;
#endif
    (void) port;

    std::vector<teca_metadata> up_reqs;

#if defined(TECA_HAS_MPI)
    int rank = 0;
    int n_ranks = 1;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
        MPI_Comm_size(this->get_communicator(), &n_ranks);
    }
#endif

    // get the executive control keys for the upstream.
    const teca_metadata &md_in = input_md[0];

    std::string up_initializer_key;
    if (md_in.get("index_initializer_key", up_initializer_key))
    {
        TECA_ERROR("Failed to locate index_initializer_key")
        return up_reqs;
    }

    std::string up_request_key;
    if (md_in.get("index_request_key", up_request_key))
    {
        TECA_ERROR("Failed to locate index_request_key")
        return up_reqs;
    }

    long n_indices_up = 0;
    if (md_in.get(up_initializer_key, n_indices_up))
    {
        TECA_ERROR("Missing index initializer \"" << up_initializer_key << "\"")
        return up_reqs;
    }

    // deal with subsetting of time steps
    long idx_0 = this->first_step < 0 ? 0 : this->first_step;
    long idx_1 = this->last_step < 1 ? n_indices_up - 1 : this->last_step;
    n_indices_up = idx_1 - idx_0 + 1;

    // estimate the number of files
    long n_steps_left = n_indices_up % this->steps_per_file;
    long n_files = n_indices_up / this->steps_per_file;

    // get the file id requested of us, convert this into a set of
    // upstream indices
    long file_id;
    if (request.get("file_id", file_id))
    {
        TECA_ERROR("failed to determiine requested file id")
        return up_reqs;
    }

    long first_index = idx_0 + file_id*this->steps_per_file;
    long n_indices = ((file_id == n_files) && n_steps_left ?
        n_steps_left : this->steps_per_file);

    // construct the base request, pass through incoming request for bounds,
    // arrays, etc...  reset executive control keys
    teca_metadata base_req(request);
    base_req.remove("file_id");
    base_req.set("index_request_key", up_request_key);

    // initialize the set of requests needed to write the requested file
    up_reqs.resize(n_indices, base_req);

    // fix the indices
    for (long i = 0; i < n_indices; ++i)
        up_reqs[i].set(up_request_key, first_index + i);


    // the upstream requests are all queued up. Before issuing them
    // intialize file specific book keeping structure
    teca_cf_writer::internals_t::time_series_layout layout;
    if (layout.create(file_id, first_index, n_indices, this->file_name,
        this->date_format, md_in, this->mode_flags, this->use_unlimited_dim))
    {
        TECA_ERROR("Failed to create the file for id " << file_id)
        up_reqs.clear();
        return up_reqs;
    }
    this->internals->file_table[file_id] = std::move(layout);

    return up_reqs;
}



// --------------------------------------------------------------------------
const_p_teca_dataset teca_cf_writer::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request, int streaming)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_cf_writer::execute" << std::endl;
#endif
    (void)port;

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int n_ranks = 1;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
        MPI_Comm_size(this->get_communicator(), &n_ranks);
    }
#endif

    // we will be presented with a collection of datasets that are to be
    // placed in a single file. this file is identified by the incoming
    // request file_id.

    // get the file id requested of us
    long file_id = 0;
    if (request.get("file_id", file_id))
    {
        TECA_ERROR("failed to determine requested file id")
        return nullptr;
    }

    // get the handle associated with this file
    teca_cf_writer::internals_t::file_table_iterator_t it =
        this->internals->file_table.find(file_id);

    if (it == this->internals->file_table.end())
    {
        TECA_ERROR("failed to locate a handle for file id " << file_id)
        return nullptr;
    }

    teca_cf_writer::internals_t::time_series_layout &layout = it->second;
    if (!layout.opened())
    {
        TECA_ERROR("handle for file id " << file_id << " is not open")
        return nullptr;
    }

    // get the number of datasets. these will be written to the file
    long n_indices = input_data.size();

    // set up the write. collect various data and metadata
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        if (rank == 0)
            TECA_ERROR("input mesh 0 is empty input or not a cartesian mesh")
        return nullptr;
    }

    const_p_teca_variant_array x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array z = in_mesh->get_z_coordinates();

    std::string x_variable;
    std::string y_variable;
    std::string z_variable;
    std::string t_variable;

    in_mesh->get_x_coordinate_variable(x_variable);
    in_mesh->get_y_coordinate_variable(y_variable);
    in_mesh->get_z_coordinate_variable(z_variable);
    in_mesh->get_t_coordinate_variable(t_variable);

    // collect the set of arrays to write
    std::vector<long> req_ids(n_indices);
    std::vector<const_p_teca_array_collection> point_arrays(n_indices);
    std::vector<const_p_teca_array_collection> info_arrays(n_indices);
    for (long i = 0; i < n_indices; ++i)
    {
        // convert to cartesian mesh
        in_mesh = std::dynamic_pointer_cast
            <const teca_cartesian_mesh>(input_data[i]);

        if (!in_mesh)
        {
            // rank 0 should always have data, but after a reduction only rank
            // 0 would have data
            if (rank == 0)
                TECA_ERROR("input mesh " << i
                    << " is empty input or not a cartesian mesh")
            return nullptr;
        }

        // get the collection of arrays
        point_arrays[i] = in_mesh->get_point_arrays();
        info_arrays[i] = in_mesh->get_information_arrays();

        if (in_mesh->get_request_index(req_ids[i]))
        {
            TECA_ERROR("failed to get the request index from mesh " << i)
            return nullptr;
        }
    }

    // define the file layout the first time through
    if (!layout.defined() && layout.define(x, y, z, x_variable,
        y_variable, z_variable, point_arrays, info_arrays,
        this->compression_level, input_data[0]->get_metadata()))
    {
        TECA_ERROR("failed to define file " << file_id)
        return nullptr;
    }


    // write the data
    if (layout.write(req_ids, point_arrays, info_arrays))
    {
        TECA_ERROR("Failed to write " << file_id)
        return nullptr;
    }

    // close the file when all data has been written
    if (!streaming)
    {
        if (!layout.completed())
            TECA_ERROR("File " << file_id << " is incomplete")

        layout.close();
        this->internals->file_table.erase(it);
    }

    /*
    // if there is only 1 input datatset, pass the dataset through
    // there are more than 1 return a nullptr. TODO -- how best to
    // return multiple datasets? teca_database can do it, but downstream
    // may not understand
    p_teca_dataset out_mesh;
    if (n_in_indices == 1)
    {
        out_mesh = in_mesh->new_instance();
        out_mesh->shallow_copy(
            std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));
    }
    */

    return nullptr;
}
