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

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


class teca_cf_writer_internals
{
public:
    // creates and writes a NetCDF dataset from the passed in data.
    // data is organized into a vector of array collections, one per
    // time step.
    static
    int write(const std::string &file_name, int mode, int use_unlimited_dim,
        const std::vector<long> &request_ids, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        const const_p_teca_variant_array &t, const std::string &x_variable,
        const std::string &y_variable, const std::string &z_variable,
        const std::string &t_variable, const teca_metadata &array_attributes,
        const std::vector<const_p_teca_array_collection> &arrays);
};

// --------------------------------------------------------------------------
int teca_cf_writer_internals::write(const std::string &file_name, int mode,
    int use_unlimited_dim, const std::vector<long> &request_ids,
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, const const_p_teca_variant_array &t,
    const std::string &x_variable, const std::string &y_variable,
    const std::string &z_variable, const std::string &t_variable,
    const teca_metadata &array_attributes,
    const std::vector<const_p_teca_array_collection> &arrays)
{
    (void) request_ids;

    int ierr = NC_NOERR;

    // create the output file
    teca_netcdf_util::netcdf_handle fh;
    if (fh.create(file_name.c_str(), mode))
    {
        TECA_ERROR("failed to create file \"" << file_name << "\"")
        return -1;
    }

    // files are always written in 4D. at least one of the coordinates must
    // have data for the others construct a length one array containing zeros
    const_p_teca_variant_array coord_array
        = x ? x : y ? y : z ? z : t ? t : nullptr;

    if (!coord_array)
    {
        TECA_ERROR("invalid coordinates")
        return -1;
    }

    const_p_teca_variant_array coord_arrays[4];
    std::string coord_array_names[4];
    size_t dims[4] = {0, 0, 0, 0};


    // the cf reader always creates 4D data, but some other tools choke
    // on it, notably ParView. All dimensions of 1 are safe to skip, unless
    // we are writing a variable with 1 value.
    unsigned long skip_dim_of_1 = (x && x->size() > 1 ? 1 : 0) +
        (y && y->size() > 1 ? 1 : 0) + (z  && z->size() > 1 ? 1 : 0);

    int n_dims = 0;
    if (t)
    {
        coord_arrays[n_dims] = t;
        coord_array_names[n_dims] = t_variable.empty() ? "time" : t_variable;
        dims[n_dims] = use_unlimited_dim ? NC_UNLIMITED : t->size();
        ++n_dims;
    }
    if (z)
    {
        if (!skip_dim_of_1 || z->size() > 1)
        {
            coord_arrays[n_dims] = z;
            coord_array_names[n_dims] = z_variable.empty() ? "z" : z_variable;
            dims[n_dims] = n_dims == 0 && use_unlimited_dim ? NC_UNLIMITED : z->size();
            ++n_dims;
        }
    }
    if (y)
    {
        if (!skip_dim_of_1 || y->size() > 1)
        {
            coord_arrays[n_dims] = y;
            coord_array_names[n_dims] = y_variable.empty() ? "y" : y_variable;
            dims[n_dims] = n_dims == 0 && use_unlimited_dim ? NC_UNLIMITED : y->size();
            ++n_dims;
        }
    }
    if (x)
    {
        if (!skip_dim_of_1 || x->size() > 1)
        {
            coord_arrays[n_dims] = x;
            coord_array_names[n_dims] = x_variable.empty() ? "x" : x_variable;
            dims[n_dims] = n_dims == 0 && use_unlimited_dim ? NC_UNLIMITED : x->size();
            ++n_dims;
        }
    }

    // dictionary of names to ncids
    int dim_ids[4] = {-1};
    std::map<std::string, int> var_ids;

    for (int i = 0; i < n_dims; ++i)
    {
        // define dimension
        int dim_id = -1;
        if ((ierr = nc_def_dim(fh.get(), coord_array_names[i].c_str(), dims[i], &dim_id)) != NC_NOERR)
        {
            TECA_ERROR("failed to define dimensions for coordinate axis "
                <<  i << " \"" << coord_array_names[i] << "\" " << nc_strerror(ierr))
            return -1;
        }

        // save the dim id
        dim_ids[i] = dim_id;

        // define variable for the axis
        int var_id = -1;
        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),
            int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
            if ((ierr = nc_def_var(fh.get(), coord_array_names[i].c_str(), type, 1, &dim_id, &var_id)) != NC_NOERR)
            {
                TECA_ERROR("failed to define variables for coordinate axis "
                    <<  i << " \"" << coord_array_names[i] << "\" " << nc_strerror(ierr))
                return -1;
            }
            )

        // save the var id
        var_ids[coord_array_names[i]] = var_id;
    }

    // define variables for each point array
    if (arrays.size())
    {
        unsigned int n_arrays = arrays[0]->size();
        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            std::string name = arrays[0]->get_name(i);
            const_p_teca_variant_array array = arrays[0]->get(i);

            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                array.get(),
                // define variable
                int var_id = -1;
                int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
                if ((ierr = nc_def_var(fh.get(), name.c_str(), type, n_dims, dim_ids, &var_id)) != NC_NOERR)
                {
                    TECA_ERROR("failed to define variable for point array \"" << name << "\". "
                        << nc_strerror(ierr))
                    return -1;
                }
                // save the var id
                var_ids[name] = var_id;
                )
        }
    }

    // write attributes of the varibles in hand
    std::map<std::string,int>::iterator it = var_ids.begin();
    std::map<std::string,int>::iterator end = var_ids.end();
    for (; it != end; ++it)
    {
        teca_metadata array_atts;
        std::string array_name = it->first;
        if (array_attributes.get(array_name, array_atts))
        {
            // It's ok for a variable not to have attributes
            //TECA_WARNING("No array attributes for \"" << array_name << "\"")
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
            if ((att_name == "id") || (att_name == "dims") || (att_name == "dim_names") ||
                (att_name == "type") || (att_name == "centering"))
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
                if ((ierr = nc_put_att_text(fh.get(), var_id, att_name.c_str(), att_val.size()+1,
                    att_val.c_str())) != NC_NOERR)
                {
                    TECA_ERROR("failed to put attribute \"" << att_name << "\"")
                }
                continue;
                )

            // handle POD types
            TEMPLATE_DISPATCH(const teca_variant_array_impl,
                att_values.get(),

                int type = teca_netcdf_util::netcdf_tt<NT>::type_code;
                const NT *pvals = static_cast<TT*>(att_values.get())->get();
                unsigned long n_vals = att_values->size();

                if ((ierr = nc_put_att(fh.get(), var_id, att_name.c_str(), type,
                    n_vals, pvals)) != NC_NOERR)
                {
                    TECA_ERROR("failed to put attribute \"" << att_name << "\" "
                        << nc_strerror(ierr))
                }
                )
        }
    }

    // end metadata definition phase
    ierr = nc_enddef(fh.get());

    // write the coordinate arrays
    for (int i = 0; i < n_dims; ++i)
    {
        // look up the var id
        std::string array_name = coord_array_names[i];
        std::map<std::string, int>::iterator it = var_ids.find(array_name);
        if (it  == var_ids.end())
        {
            TECA_ERROR("No var id for \"" << array_name << "\"")
            return -1;
        }
        int var_id = it->second;

        size_t start = 0;
        size_t count = dims[i];

        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            coord_arrays[i].get(),
            const NT *pa = static_cast<TT*>(coord_arrays[i].get())->get();
            if ((ierr = nc_put_vara(fh.get(), var_id, &start, &count, pa)) != NC_NOERR)
            {
                TECA_ERROR("failed to write \"" << coord_array_names[i] << "\" axis. "
                    << nc_strerror(ierr))
                return -1;
            }
            )
    }

    // write point arrays
    if (arrays.size())
    {
        unsigned int n_steps = t->size();
        size_t starts[4] = {0, 0, 0, 0};
        size_t counts[4] = {1, 0, 0, 0};
        for (int i = 1; i < n_dims; ++i)
            counts[i] = dims[i];

        for (unsigned int q = 0; q < n_steps; ++q)
        {
            unsigned int n_arrays = arrays[q]->size();
            for (unsigned int i = 0; i < n_arrays; ++i)
            {
                std::string array_name = arrays[q]->get_name(i);
                const_p_teca_variant_array array = arrays[q]->get(i);

                // look up the var id
                std::map<std::string, int>::iterator it = var_ids.find(array_name);
                if (it  == var_ids.end())
                {
                    TECA_ERROR("No var id for \"" << array_name << "\"")
                    return -1;
                }
                int var_id = it->second;

                TEMPLATE_DISPATCH(const teca_variant_array_impl,
                    array.get(),
                    const NT *pa = static_cast<TT*>(array.get())->get();
                    if ((ierr = nc_put_vara(fh.get(), var_id, starts, counts, pa)) != NC_NOERR)
                    {
                        TECA_ERROR("failed to write array \"" << array_name << "\". "
                            << nc_strerror(ierr))
                        return -1;
                    }
                    )
            }
            starts[0] += 1;
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
teca_cf_writer::teca_cf_writer() :
    file_name(""), date_format("%F-%HZ"), steps_per_file(8),
    mode_flags(NC_CLOBBER|NC_NETCDF4), use_unlimited_dim(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cf_writer::~teca_cf_writer()
{}

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
        TECA_POPTS_GET(unsigned int, prefix, steps_per_file,
            "set the number of time steps to write per file (8)")
        TECA_POPTS_GET(int, prefix, mode_flags,
            "mode flags to pass to NetCDF when creating the file (NC_CLOBBER)")
        TECA_POPTS_GET(int, prefix, use_unlimited_dim,
            "if set the slowest varying dimension is specified to be "
            "NC_UNLIMITED. (1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cf_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, std::string, prefix, date_format)
    TECA_POPTS_SET(opts, unsigned int, prefix, steps_per_file)
    TECA_POPTS_SET(opts, int, prefix, mode_flags)
    TECA_POPTS_SET(opts, int, prefix, use_unlimited_dim)
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

    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
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

    // estimate the number of files we create for this run
    long n_files = n_indices/this->steps_per_file;

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

    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
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

    // estimate the number of files
    long n_large_files = n_indices_up % this->steps_per_file;

    // get the file id requested of us, convert this into a set of
    // upstream indices
    long file_id;
    if (request.get("file_id", file_id))
    {
        TECA_ERROR("failed to determiine requested file id")
        return up_reqs;
    }

    long first_index = file_id < n_large_files ?
        file_id*(this->steps_per_file + 1) :
        file_id*this->steps_per_file + n_large_files;

    long n_indices = this->steps_per_file +
        (file_id < n_large_files ? 1 : 0);

    // construct the base request, pass through incoming request for bounds,
    // arrays, etc...  reset executive control keys
    teca_metadata base_req(request);
    base_req.remove("file_id");
    base_req.set("index_request_key", up_request_key);

    // initialize the set of requests needed to write the requested file
    up_reqs.resize(n_indices, base_req);

    // fix the index
    for (long i = 0; i < n_indices; ++i)
        up_reqs[i].set(up_request_key, first_index + i);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cf_writer::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_cf_writer::execute" << std::endl;
#endif
    (void)port;

    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
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
        TECA_ERROR("failed to determiine requested file id")
        return nullptr;
    }

    // get the number of datasets. these will be written to the file
    long n_in_indices = input_data.size();

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

    // get the attributes
    teca_metadata in_atts;
    if (in_mesh->get_metadata().get("attributes", in_atts))
    {
        TECA_ERROR("failed to get attribute metadata")
        return nullptr;
    }

    // get the executive control keys
    std::string request_key;
    if (in_mesh->get_metadata().get("index_request_key", request_key))
    {
        TECA_ERROR("Input metadata on mesh 0 is missing index_request_key")
        return nullptr;
    }

    // re-construct the time axis
    p_teca_double_array t = teca_double_array::New(n_in_indices);
    double *pt = t->get();

    // collect the set of arrays to write
    std::vector<long> req_ids(n_in_indices);
    std::vector<const_p_teca_array_collection> arrays(n_in_indices);
    for (long i = 0; i < n_in_indices; ++i)
    {
        // convert to cartesian mesh
        in_mesh = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[i]);

        if (!in_mesh)
        {
            if (rank == 0)
                TECA_ERROR("input mesh " << i << " is empty input or not a cartesian mesh")
            return nullptr;
        }

        // get the time
        if (in_mesh->get_time(*(pt+i)))
        {
            TECA_ERROR("input mesh " << i << " is missing time metadata")
            return nullptr;
        }

        // get the collection of arrays
        arrays[i] = in_mesh->get_point_arrays();

        // get the request id
        if (in_mesh->get_metadata().get(request_key, req_ids[i]))
        {
            TECA_ERROR("failed to get the request index \""
                << request_key << "\" from mesh " << i)
            return nullptr;
        }
    }

    // construct the file name
    std::string out_file = this->file_name;
    if (!this->date_format.empty())
    {
        // use the date string for the time information in the filename
        // get the calendar and units information
        std::string calendar;
        std::string units;
        in_mesh->get_calendar(calendar);
        in_mesh->get_time_units(units);

        if (calendar.empty() || units.empty())
        {
            // no calendar metadata, fallback to time step
            TECA_WARNING("input dataset is missing calendaring metadata")
            teca_file_util::replace_timestep(out_file, file_id);
        }
        else if (teca_file_util::replace_time(out_file, *pt,
            calendar, units, this->date_format))
        {
            // conversion failed, fall back to time step
            TECA_WARNING("failed to convert relative time value \"" << *pt
                << "\" to with the calendar \"" << calendar << "\" units \""
                << units << "\" and format \"" << this->date_format << "\".")
            teca_file_util::replace_timestep(out_file, file_id);
        }
    }
    else
    {
        // use the timestep for the time information in the filename
        teca_file_util::replace_timestep(out_file, file_id);
    }

    // replace extension
    teca_file_util::replace_extension(out_file, "nc");

    // write the data
    if (teca_cf_writer_internals::write(out_file, this->mode_flags,
        this->use_unlimited_dim, req_ids, x, y, z, t, x_variable, y_variable,
        z_variable, t_variable, in_atts, arrays))
    {
        TECA_ERROR("Failed to write \"" << out_file << "\"")
        return nullptr;
    }

    return nullptr;
}
