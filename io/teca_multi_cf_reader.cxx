#include "teca_multi_cf_reader.h"
#include "teca_array_attributes.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh.h"
#include "teca_cf_reader.h"
#include "teca_array_collection.h"
#include "teca_programmable_algorithm.h"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <map>
#include <set>
#include <utility>
#include <memory>
#include <iomanip>
#include <unistd.h>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

// internals for the cf reader
class teca_multi_cf_reader_internals
{
public:
    teca_multi_cf_reader_internals()
    {}

    // read a subset of arrays using the passed in reader. the passed
    // request defines what is read except that only the passed in arrays
    // will be read. the resulting data is pointed to by mesh_out.
    // returns 0 if successful.
    static
    int read_arrays(p_teca_cf_reader reader,
        const teca_metadata &request,
        const std::vector<std::string> &arrays,
        p_teca_cartesian_mesh &mesh_out);

public:
    // a container that packages informatiuon associated with a reader
    struct cf_reader_instance
    {
        cf_reader_instance(const p_teca_cf_reader r,
            const std::set<std::string> v) : reader(r), variables(v) {}

        p_teca_cf_reader reader;            // the reader
        teca_metadata metadata;             // cached metadata
        std::set<std::string> variables;    // variables to read
    };

    using p_cf_reader_instance = std::shared_ptr<cf_reader_instance>;

    teca_metadata metadata;     // cached aglomerated metadata
    std::string time_reader;    // names the reader that provides time axis
    std::string geometry_reader;// names the reader the provides mesh geometry

    using reader_map_t = std::map<std::string, p_cf_reader_instance>;
    reader_map_t readers;
};

// --------------------------------------------------------------------------
int teca_multi_cf_reader_internals::read_arrays(p_teca_cf_reader reader,
    const teca_metadata &request, const std::vector<std::string> &arrays,
    p_teca_cartesian_mesh &mesh_out)
{
    // this stage will inject the arrays into the request
    // and extract the mesh
    p_teca_programmable_algorithm dc = teca_programmable_algorithm::New();

    dc->set_name("reader_driver");

    dc->set_request_callback([&](unsigned int,
        const std::vector<teca_metadata> &,
        const teca_metadata &) -> std::vector<teca_metadata>
        {
            teca_metadata req(request);
            req.set("arrays", arrays);
            return {req};
        });

    dc->set_execute_callback([&](unsigned int,
        const std::vector<const_p_teca_dataset> &data_in,
        const teca_metadata &) -> const_p_teca_dataset
        {
            if (data_in.size())
                mesh_out = std::dynamic_pointer_cast<teca_cartesian_mesh>(
                    std::const_pointer_cast<teca_dataset>(data_in[0]));
            return nullptr;
        });

    dc->set_input_connection(reader->get_output_port());

    // read the data
    dc->update();

    // check for errors
    if (!mesh_out)
        return -1;

    return 0;
}



// --------------------------------------------------------------------------
teca_multi_cf_reader::teca_multi_cf_reader() :
    x_axis_variable("lon"),
    y_axis_variable("lat"),
    z_axis_variable(""),
    t_axis_variable("time"),
    t_calendar(""),
    t_units(""),
    filename_time_template(""),
    periodic_in_x(0),
    periodic_in_y(0),
    periodic_in_z(0),
    thread_pool_size(-1),
    internals(new teca_multi_cf_reader_internals)
{}

// --------------------------------------------------------------------------
teca_multi_cf_reader::~teca_multi_cf_reader()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_multi_cf_reader::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_multi_cf_reader":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, metadata_cache_dir,
            "a directory where metadata caches can be stored ()")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "name of variable that has x axis coordinates (lon)")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "name of variable that has y axis coordinates (lat)")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "name of variable that has z axis coordinates ()")
        TECA_POPTS_GET(std::string, prefix, t_axis_variable,
            "name of variable that has t axis coordinates (time)")
        TECA_POPTS_GET(std::string, prefix, t_calendar,
            "name of variable that has the time calendar (calendar)")
        TECA_POPTS_GET(std::string, prefix, t_units,
            "a std::get_time template for decoding time from the input filename")
        TECA_POPTS_GET(std::string, prefix, filename_time_template,
            "name of variable that has the time unit (units)")
        TECA_POPTS_GET(std::vector<double>, prefix, t_values,
            "name of variable that has t axis values set by the"
            "the user if the file doesn't have time variable set ()")
        TECA_POPTS_GET(int, prefix, periodic_in_x,
            "the dataset has apriodic boundary in the x direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_y,
            "the dataset has apriodic boundary in the y direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_z,
            "the dataset has apriodic boundary in the z direction (0)")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "set the number of I/O threads (-1)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, metadata_cache_dir)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_calendar)
    TECA_POPTS_SET(opts, std::string, prefix, t_units)
    TECA_POPTS_SET(opts, std::string, prefix, filename_time_template)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, t_values)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_x)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_y)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_z)
    TECA_POPTS_SET(opts, int, prefix, thread_pool_size)
}
#endif

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::clear_cached_metadata()
{
    this->internals->metadata.clear();
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::add_reader(const std::string &key,
    const std::string &files_regex,
    int provides_time, int provides_geometry,
    const std::vector<std::string> &variables)
{
    p_teca_cf_reader reader = teca_cf_reader::New();
    reader->set_files_regex(files_regex);

    this->internals->readers[key] =
        teca_multi_cf_reader_internals::p_cf_reader_instance(
            new teca_multi_cf_reader_internals::cf_reader_instance(
                reader, std::set<std::string>(variables.begin(), variables.end())));

    if (provides_time)
        this->internals->time_reader = key;

    if (provides_geometry)
        this->internals->geometry_reader = key;

    return 0;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::set_time_reader(const std::string &key)
{
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(key);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader associated with \"" << key << "\"")
        return -1;
    }

    this->internals->time_reader = key;
    return 0;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::set_geometry_reader(const std::string &key)
{
    teca_multi_cf_reader_internals::reader_map_t::iterator it = this->internals->readers.find(key);
    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader associated with \"" << key << "\"")
        return -1;
    }

    this->internals->geometry_reader = key;
    return 0;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::add_variable_reader(const std::string &key,
    const std::string &variable)
{
    teca_multi_cf_reader_internals::reader_map_t::iterator it = this->internals->readers.find(key);
    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader associated with \"" << key << "\"")
        return -1;
    }

    it->second->variables.insert(variable);
    return 0;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::set_variable_reader(const std::string &key,
    const std::vector<std::string> &variables)
{
    teca_multi_cf_reader_internals::reader_map_t::iterator it = this->internals->readers.find(key);
    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader associated with \"" << key << "\"")
        return -1;
    }

    it->second->variables.insert(variables.begin(), variables.end());
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_multi_cf_reader::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_multi_cf_reader::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    // return cached metadata. cache is cleared if
    // any of the algorithms properties are modified
    if (this->internals->metadata)
        return this->internals->metadata;

    teca_metadata atts_out;
    teca_metadata coords_out;
    std::vector<std::string> vars_out;

    // update the metadata for the managed readers
    teca_multi_cf_reader_internals::reader_map_t::iterator it = this->internals->readers.begin();
    teca_multi_cf_reader_internals::reader_map_t::iterator end = this->internals->readers.end();
    for (; it != end; ++it)
    {
        const std::string &key = it->first;
        teca_multi_cf_reader_internals::p_cf_reader_instance &inst = it->second;

        // pass run time control parameters
        inst->reader->set_x_axis_variable(this->x_axis_variable);
        inst->reader->set_y_axis_variable(this->y_axis_variable);
        inst->reader->set_z_axis_variable(this->z_axis_variable);
        inst->reader->set_t_axis_variable(this->t_axis_variable);
        inst->reader->set_t_calendar(this->t_calendar);
        inst->reader->set_t_units(this->t_units);
        inst->reader->set_filename_time_template(this->filename_time_template);
        inst->reader->set_periodic_in_x(this->periodic_in_x);
        inst->reader->set_periodic_in_y(this->periodic_in_y);
        inst->reader->set_periodic_in_z(this->periodic_in_z);
        inst->reader->set_metadata_cache_dir(this->metadata_cache_dir);
        inst->reader->set_thread_pool_size(this->thread_pool_size);

        // update the internal reader's metadata
        inst->metadata = inst->reader->update_metadata();

        // grab coordinates and attributes
        teca_metadata atts_in;
        inst->metadata.get("attributes", atts_in);

        teca_metadata coords_in;
        inst->metadata.get("coordinates", coords_in);

        if (key == this->internals->time_reader)
        {
            //pass time axis and attributes
            std::string t_variable;
            if (coords_in.get("t_variable", t_variable))
            {
                TECA_ERROR("Failed to get the time varaible name")
                return teca_metadata();
            }
            coords_out.set("t_variable", t_variable);

            teca_metadata t_atts;
            if (atts_in.get(t_variable, t_atts))
            {
                TECA_ERROR("Failed to get attributes for \""
                    << t_variable << "\"")
                return teca_metadata();
            }
            atts_out.set(t_variable, t_atts);

            p_teca_variant_array t = coords_in.get("t");
            if (!t)
            {
                TECA_ERROR("Failed to get the time axis")
                return teca_metadata();
            }
            coords_out.set("t", t);

            // pass pipeline control keys
            std::string initializer_key;
            if (inst->metadata.get("index_initializer_key", initializer_key))
            {
                TECA_ERROR("Failed to get the index_initializer_key")
                return teca_metadata();
            }
            this->internals->metadata.set("index_initializer_key", initializer_key);

            std::string request_key;
            if (inst->metadata.get("index_request_key", request_key))
            {
                TECA_ERROR("Failed to get the index_request_key")
                return teca_metadata();
            }
            this->internals->metadata.set("index_request_key", request_key);

            long n_indices = 0;
            if (inst->metadata.get(initializer_key, n_indices))
            {
                TECA_ERROR("Failed to get the value of the intitializer \""
                    << initializer_key << "\"")
                return teca_metadata();
            }
            this->internals->metadata.set(initializer_key, n_indices);
        }

        if (key == this->internals->geometry_reader)
        {
            // pass x axis and attributes
            std::string x_variable;
            if (coords_in.get("x_variable", x_variable))
            {
                TECA_ERROR("Failed to get the x axis varaible name")
                return teca_metadata();
            }
            coords_out.set("x_variable", x_variable);

            if (!this->x_axis_variable.empty())
            {
                teca_metadata x_atts;
                if (atts_in.get(x_variable, x_atts))
                {
                    TECA_ERROR("Failed to get attributes for \""
                        << x_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(x_variable, x_atts);
            }

            p_teca_variant_array x = coords_in.get("x");
            if (!x)
            {
                TECA_ERROR("Failed to get the y axis")
                return teca_metadata();
            }
            coords_out.set("x", x);

            // pass x axis and attributes
            std::string y_variable;
            if (coords_in.get("y_variable", y_variable))
            {
                TECA_ERROR("Failed to get the x axis varaible name")
                return teca_metadata();
            }
            coords_out.set("y_variable", y_variable);

            if (!this->y_axis_variable.empty())
            {
                teca_metadata y_atts;
                if (atts_in.get(y_variable, y_atts))
                {
                    TECA_ERROR("Failed to get attributes for \""
                        << y_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(y_variable, y_atts);
            }

            p_teca_variant_array y = coords_in.get("y");
            if (!y)
            {
                TECA_ERROR("Failed to get the x axis")
                return teca_metadata();
            }
            coords_out.set("y", y);

            // pass y axis and attributes
            std::string z_variable;
            if (coords_in.get("z_variable", z_variable))
            {
                TECA_ERROR("Failed to get the y axis varaible name")
                return teca_metadata();
            }
            coords_out.set("z_variable", z_variable);

            if (!this->z_axis_variable.empty())
            {
                teca_metadata z_atts;
                if (atts_in.get(z_variable, z_atts))
                {
                    TECA_ERROR("Failed to get attributes for \""
                        << z_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(z_variable, z_atts);
            }

            p_teca_variant_array z = coords_in.get("z");
            if (!z)
            {
                TECA_ERROR("Failed to get the z axis")
                return teca_metadata();
            }
            coords_out.set("z", z);

            // pass periodicity
            coords_out.set("periodic_in_x", this->periodic_in_x);
            coords_out.set("periodic_in_y", this->periodic_in_y);
            coords_out.set("periodic_in_z", this->periodic_in_z);

            // pass bounds
            p_teca_variant_array bounds = inst->metadata.get("bounds");
            if (!bounds)
            {
                TECA_ERROR("Failed to get the mesh bounds")
                return teca_metadata();
            }
            this->internals->metadata.set("bounds", bounds);

            // pass whole_extent
            p_teca_variant_array whole_extent = inst->metadata.get("whole_extent");
            if (!whole_extent)
            {
                TECA_ERROR("Failed to get the mesh whole_extent")
                return teca_metadata();
            }
            this->internals->metadata.set("whole_extent", whole_extent);
        }

        // pass variable attributes
        std::set<std::string>::iterator it = inst->variables.begin();
        std::set<std::string>::iterator end = inst->variables.end();
        for (; it != end; ++it)
        {
            const std::string &var_name = *it;

            // add it to the list of variables exposed down stream
            vars_out.push_back(var_name);

            teca_metadata var_atts;
            if (atts_in.get(var_name, var_atts))
            {
                TECA_ERROR("Failed to get attributes for \""
                    << var_name << "\"")
                return teca_metadata();
            }
            atts_out.set(var_name, var_atts);
        }
    }

    // set up the output
    this->internals->metadata.set("variables", vars_out);
    this->internals->metadata.set("attributes", atts_out);
    this->internals->metadata.set("coordinates", coords_out);

    return this->internals->metadata;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_multi_cf_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_multi_cf_reader::execute" << endl;
#endif
    (void)port;
    (void)input_data;

    // get the requested arrays
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);
    size_t n_arrays = req_arrays.size();

    // route the requested arrays to the correct reader.
    using array_router_t = std::map<std::string, std::vector<std::string>>;
    using reader_map_it_t = teca_multi_cf_reader_internals::reader_map_t::iterator;

    array_router_t array_router;

    reader_map_it_t it = this->internals->readers.begin();
    reader_map_it_t end = this->internals->readers.end();
    for (; it != end; ++it)
    {
        const std::string &key = it->first;
        teca_multi_cf_reader_internals::p_cf_reader_instance &inst = it->second;

        for (size_t i = 0; i < n_arrays; ++i)
        {
            const std::string &array = req_arrays[i];
            std::set<std::string>::iterator vit = inst->variables.find(array);
            if (vit != inst->variables.end())
                array_router[key].push_back(*vit);
        }
    }

    // read mesh geometry first
    p_teca_cartesian_mesh mesh_out;
    if (teca_multi_cf_reader_internals::read_arrays(
        this->internals->readers[this->internals->geometry_reader]->reader,
        request, array_router[this->internals->geometry_reader], mesh_out))
    {
        TECA_ERROR("Failed to read mesh geometry")
        return nullptr;
    }

    // read the rest of the arrays
    it = this->internals->readers.begin();
    for (; it != end; ++it)
    {
        const std::string &key = it->first;
        teca_multi_cf_reader_internals::p_cf_reader_instance &inst = it->second;

        // skip the geometry reader, we already read those above
        if (key == this->internals->geometry_reader)
            continue;

        // read the reader's arrays
        p_teca_cartesian_mesh tmp;
        if (teca_multi_cf_reader_internals::read_arrays(inst->reader,
            request, array_router[key], tmp))
        {
            TECA_ERROR("Failed to read mesh geometry")
            return nullptr;
        }

        // pass them into the output
        if (mesh_out->get_point_arrays()->append(tmp->get_point_arrays()) ||
            mesh_out->get_cell_arrays()->append(tmp->get_cell_arrays()) ||
            mesh_out->get_x_edge_arrays()->append(tmp->get_x_edge_arrays()) ||
            mesh_out->get_y_edge_arrays()->append(tmp->get_y_edge_arrays()) ||
            mesh_out->get_z_edge_arrays()->append(tmp->get_z_edge_arrays()) ||
            mesh_out->get_x_face_arrays()->append(tmp->get_x_face_arrays()) ||
            mesh_out->get_y_face_arrays()->append(tmp->get_y_face_arrays()) ||
            mesh_out->get_z_face_arrays()->append(tmp->get_z_face_arrays()) ||
            mesh_out->get_information_arrays()->append(
                tmp->get_information_arrays()))
        {
            TECA_ERROR("Failed to pass the arrays")
            return nullptr;
        }
    }

    return mesh_out;
}
