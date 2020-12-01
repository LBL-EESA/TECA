#include "teca_multi_cf_reader.h"
#include "teca_array_attributes.h"
#include "teca_file_util.h"
#include "teca_string_util.h"
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

    static
    int parse_cf_reader_section(teca_file_util::line_buffer &lines,
        std::string &name, std::string &regex, int &provides_time,
        int &provides_geometry, std::vector<std::string> &variables);

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
int teca_multi_cf_reader_internals::parse_cf_reader_section(
    teca_file_util::line_buffer &lines, std::string &name,
    std::string &regex, int &provides_time, int &provides_geometry,
    std::vector<std::string> &variables)
{
    name = "";
    regex = "";
    provides_time = 0;
    provides_geometry = 0;
    variables.clear();

    // the section was valid if at least regex and vars were found
    int have_name = 0;
    int have_regex = 0;
    int have_variables = 0;
    int have_time_reader = 0;
    int have_geometry_reader = 0;

    // the caller is expected to pass the current line and that line
    // is expected to be "[cf_reader]".
    char *l = lines.current();
    teca_string_util::skip_pad(l);

    if (l[0] == '[')
        lines.pop();

    while (lines)
    {
        long lno = lines.line_number() + 1;
        l = lines.current();

        // stop at the next section header
        teca_string_util::skip_pad(l);
        if (l[0] == '[')
            break;

        lines.pop();

        // skip comments
        if (teca_string_util::is_comment(l))
            continue;


        // look for and process key words
        if (strncmp("name", l, 5) == 0)
        {
            if (have_name)
            {
                TECA_ERROR("Duplicate name lable found on line " << lno)
                return -1;
            }

            if (teca_string_util::extract_value<std::string>(l, name))
            {
                TECA_ERROR("Syntax error when parsing name on line " << lno)
                return -1;
            }

            have_name = 1;
        }
        else if (strncmp("regex", l, 5) == 0)
        {
            if (have_regex)
            {
                TECA_ERROR("Duplicate regex lable found on line " << lno)
                return -1;
            }

            if (teca_string_util::extract_value<std::string>(l, regex))
            {
                TECA_ERROR("Syntax error when parsing regex on line " << lno)
                return -1;
            }

            have_regex = 1;
        }
        else if (strncmp("variables", l, 9) == 0)
        {
            if (have_variables)
            {
                TECA_ERROR("Duplicate regex lable found on line " << lno)
                return -1;
            }

            std::vector<char*> tmp;
            if (teca_string_util::tokenize(l, '=', tmp) || (tmp.size() != 2))
            {
                TECA_ERROR("Invalid variables specifier : \"" << l
                    << "\" on line " << lno)
                return -1;
            }

            std::vector<char*> vars;
            if (teca_string_util::tokenize(tmp[1], ',', vars) || (vars.size() < 1))
            {
                TECA_ERROR("Invalid variables specifier : \"" << l
                    << "\" on line " << lno)
                return -1;
            }

            size_t n_vars  = vars.size();
            for (size_t i = 0; i < n_vars; ++i)
            {
                char *v = vars[i];
                if (teca_string_util::skip_pad(v))
                {
                    TECA_ERROR("Invalid variable name on line " << lno)
                    return -1;
                }
                variables.push_back(v);
            }

            have_variables = 1;
        }
        else if (strncmp("provides_time", l, 11) == 0)
        {
            if (have_time_reader)
            {
                TECA_ERROR("Duplicate provides_time lable found on line " << lno)
                return -1;
            }
            provides_time = 1;
            have_time_reader = 1;
        }
        else if (strncmp("provides_geometry", l, 15) == 0)
        {
            if (have_geometry_reader)
            {
                TECA_ERROR("Duplicate provides_geometry lable found on line " << lno)
                return -1;
            }
            provides_geometry = 1;
        }
    }

    return (have_regex && have_variables) ? 0 : -1;
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
    max_metadata_ranks(1024),
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
        TECA_POPTS_GET(std::string, prefix, input_file,
            "a file dedscribing the dataset layout ()")
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
            "the dataset has a periodic boundary in the x direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_y,
            "the dataset has a periodic boundary in the y direction (0)")
        TECA_POPTS_GET(int, prefix, periodic_in_z,
            "the dataset has a periodic boundary in the z direction (0)")
        TECA_POPTS_GET(int, prefix, max_metadata_ranks,
            "set the max number of ranks for reading metadata (1024)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
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
    TECA_POPTS_SET(opts, int, prefix, max_metadata_ranks)
}
#endif

// --------------------------------------------------------------------------
int teca_multi_cf_reader::set_input_file(const std::string &input_file)
{
    int rank = 0;
    int n_ranks = 1;
#if defined(TECA_HAS_MPI)
    MPI_Comm comm = this->get_communicator();

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &n_ranks);
    }
#endif

    teca_binary_stream bs;

    // save a spot at the head of the stream for the number of readers.
    // after we know how many readers were detected we will update this.
    int num_readers = 0;
    bs.pack(num_readers);

    if (rank == 0)
    {
        teca_file_util::line_buffer lines;
        if (lines.initialize(input_file.c_str()))
        {
            TECA_ERROR("Failed to read \"" << input_file << "\"")
            return -1;
        }

        // these are global variables and need to be at the top of the file
        std::string g_data_root;
        std::string g_regex;

        while (lines)
        {
            long lno = lines.line_number();
            char *l = lines.current();
            teca_string_util::skip_pad(l);
            lines.pop();

            if (teca_string_util::is_comment(l))
                continue;

            if (strncmp("data_root", l, 9) == 0)
            {
                if (teca_string_util::extract_value<std::string>(l, g_data_root))
                {
                    TECA_ERROR("Failed to parse \"data_root\" specifier on line " << lno)
                    return -1;
                }
            }
            else if (strncmp("regex", l, 5) == 0)
            {
                if (teca_string_util::extract_value<std::string>(l, g_regex))
                {
                    TECA_ERROR("Failed to parse \"regex\" specifier on line " << lno)
                    return -1;
                }
            }
            else if (strcmp("[cf_reader]", l) == 0)
            {
                std::string name;
                std::string regex;
                int provides_time = 0;
                int provides_geometry = 0;
                std::vector<std::string> variables;

                if (teca_multi_cf_reader_internals::parse_cf_reader_section(lines,
                    name, regex, provides_time, provides_geometry, variables))
                {
                    TECA_ERROR("Failed to parse [cf_reader] section on line " << lno)
                    return -1;
                }

                // always give a name
                if (name.empty())
                    name = std::to_string(num_readers);

                // look for and replace %data_root% if it is present
                if (!g_data_root.empty())
                {
                    size_t loc = regex.find("%data_root%");
                    if (loc != std::string::npos)
                        regex.replace(loc, 11, g_data_root);
                }

                if (!g_regex.empty())
                {
                    size_t loc = regex.find("%regex%");
                    if (loc != std::string::npos)
                        regex.replace(loc, 7, g_regex);
                }

                // serialize
                bs.pack(name);
                bs.pack(regex);
                bs.pack(provides_time);
                bs.pack(provides_geometry);
                bs.pack(variables);

                num_readers += 1;
            }
        }

        // update count
        size_t ebs = bs.size();
        bs.set_write_pos(0);
        bs.pack(num_readers);
        bs.set_write_pos(ebs);
    }

    // share
    bs.broadcast(this->get_communicator());

    // deserialize
    bs.unpack(num_readers);

    if (num_readers < 1)
    {
        TECA_ERROR("No readers found in \"" << input_file << "\"")
        return -1;
    }

    int num_time_readers = 0;
    int num_geometry_readers = 0;

    for (int i = 0; i < num_readers; ++i)
    {
        std::string name;
        std::string regex;
        int provides_time;
        int provides_geometry;
        std::vector<std::string> variables;

        bs.unpack(name);
        bs.unpack(regex);
        bs.unpack(provides_time);
        bs.unpack(provides_geometry);
        bs.unpack(variables);

        num_time_readers += provides_time;
        num_geometry_readers += provides_geometry;

        if (this->add_reader(name, regex, provides_time,
            provides_geometry, variables))
        {
            TECA_ERROR("Failed to add reader " << i << " \"" << name << "\"")
            return -1;
        }
    }

    return 0;
}

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
    if (key.empty())
    {
        TECA_ERROR("Invalid key, it must not be empty")
        return -1;
    }

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
void teca_multi_cf_reader::get_variables(std::vector<std::string> &vars)
{
    vars.clear();

    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.begin();

    teca_multi_cf_reader_internals::reader_map_t::iterator end =
        this->internals->readers.end();

    for (; it != end; ++it)
    {
        vars.insert(vars.end(), it->second->variables.begin(),
            it->second->variables.end());
    }
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
        inst->reader->set_max_metadata_ranks(this->max_metadata_ranks);

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
