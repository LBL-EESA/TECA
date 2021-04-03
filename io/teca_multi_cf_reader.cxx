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
    teca_multi_cf_reader_internals() {}

    /// packages reader options
    struct cf_reader_options
    {
        cf_reader_options() :
            name(), regex(), provides_time(0), provides_geometry(0),
            variables(), x_axis_variable(), y_axis_variable(),
            z_axis_variable(), t_axis_variable(), periodic_in_x(-1),
            calendar(), t_units(), filename_time_template(),
            clamp_dimensions_of_one(-1)
            {}

        /**
         * parse one line for fields we own. if none are found
         * return 0, if one is found return 1, if an error occurs
         * return -1
         */
        int parse_line(char *line, unsigned long line_no);

        /// return the internal value if set otherwise the default
        std::string get_x_axis_variable(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        std::string get_y_axis_variable(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        std::string get_z_axis_variable(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        std::string get_t_axis_variable(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        int get_periodic_in_x(int default_val) const;

        /// return the internal value if set otherwise the default
        const std::string &get_calendar(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        const std::string &get_t_units(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        const std::string &get_filename_time_template(
            const std::string &default_val) const;

        /// return the internal value if set otherwise the default
        int get_clamp_dimensions_of_one(int default_val) const;

        /// serialize/deserialize to/from the stream
        void to_stream(teca_binary_stream &bs) const;
        void from_stream(teca_binary_stream &bs);

        std::string name;                   /// name of the reader
        std::string regex;                  /// files to serve data from
        int provides_time;                  /// set if this reader provides mesh for all
        int provides_geometry;              /// set if this reader provides time axis for all
        std::vector<std::string> variables; /// list of variables to serve
        std::string x_axis_variable;        /// name of mesh x-axis, empty to disable
        std::string y_axis_variable;        /// name of mesh y-axis, empty to disable
        std::string z_axis_variable;        /// name of mesh z-axis, empty to disable
        std::string t_axis_variable;        /// name of mesh t-axis, empty to disable
        int periodic_in_x;                  /// set to identify x-axis as periodic
        std::string calendar;               /// calendar
        std::string t_units;                /// time axis units
        std::string filename_time_template; /// for deriving time from the filename
        int clamp_dimensions_of_one;        /// ignore out of bounds requests if dim is 1
    };

    // read a subset of arrays using the passed in reader. the passed
    // request defines what is read except that only the passed in arrays
    // will be read. the resulting data is pointed to by mesh_out.
    // returns 0 if successful.
    static
    int read_arrays(p_teca_cf_reader reader,
        const teca_metadata &request,
        const std::vector<std::string> &arrays,
        p_teca_cartesian_mesh &mesh_out);


    // get configuration from a file
    static
    int parse_cf_reader_section(teca_file_util::line_buffer &lines,
        cf_reader_options &opts);

    // adds a reader to the collection
    int add_reader_instance(const cf_reader_options &options);

public:
    // a container that packages informatiuon associated with a reader
    struct cf_reader_instance
    {
        cf_reader_instance(const p_teca_cf_reader r,
            const std::set<std::string> v, const cf_reader_options &o) :
                reader(r), variables(v), options(o) {}

        p_teca_cf_reader reader;            // the reader
        teca_metadata metadata;             // cached metadata
        std::set<std::string> variables;    // variables to read
        cf_reader_options options;          // per-instance run time config
    };

    using p_cf_reader_instance = std::shared_ptr<cf_reader_instance>;

    teca_metadata metadata;            // cached aglomerated metadata
    std::string time_reader;           // names the reader that provides time axis
    std::string geometry_reader;       // names the reader the provides mesh geometry
    cf_reader_options global_options;  // default run time config

    using reader_map_t = std::map<std::string, p_cf_reader_instance>;
    reader_map_t readers;
};



// --------------------------------------------------------------------------
std::string
teca_multi_cf_reader_internals::cf_reader_options::get_x_axis_variable(
    const std::string &default_val) const
{
    if (!x_axis_variable.empty())
        return teca_string_util::emptystr(x_axis_variable);

    return default_val;
}

// --------------------------------------------------------------------------
std::string
teca_multi_cf_reader_internals::cf_reader_options::get_y_axis_variable(
    const std::string &default_val) const
{
    if (!y_axis_variable.empty())
        return teca_string_util::emptystr(y_axis_variable);

    return default_val;
}

// --------------------------------------------------------------------------
std::string
teca_multi_cf_reader_internals::cf_reader_options::get_z_axis_variable(
    const std::string &default_val) const
{
    if (!z_axis_variable.empty())
        return teca_string_util::emptystr(z_axis_variable);

    return default_val;
}

// --------------------------------------------------------------------------
std::string
teca_multi_cf_reader_internals::cf_reader_options::get_t_axis_variable(
    const std::string &default_val) const
{
    if (!t_axis_variable.empty())
        return teca_string_util::emptystr(t_axis_variable);

    return default_val;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader_internals::cf_reader_options::get_periodic_in_x(
    int default_val) const
{
    if (periodic_in_x < 0)
        return default_val;

    return periodic_in_x;
}

// --------------------------------------------------------------------------
const std::string &
teca_multi_cf_reader_internals::cf_reader_options::get_calendar(
    const std::string &default_val) const
{
    if (!calendar.empty())
        return calendar;

    return default_val;
}

// --------------------------------------------------------------------------
const std::string &
teca_multi_cf_reader_internals::cf_reader_options::get_t_units(
    const std::string &default_val) const
{
    if (!t_units.empty())
        return t_units;

    return default_val;
}

// --------------------------------------------------------------------------
const std::string &
teca_multi_cf_reader_internals::cf_reader_options::get_filename_time_template(
    const std::string &default_val) const
{
    if (!filename_time_template.empty())
        return filename_time_template;

    return default_val;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader_internals::cf_reader_options::get_clamp_dimensions_of_one(
    int default_val) const
{
    if (clamp_dimensions_of_one < 0)
        return default_val;

    return clamp_dimensions_of_one;
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader_internals::cf_reader_options::to_stream(
    teca_binary_stream &bs) const
{
    bs.pack(name);
    bs.pack(regex);
    bs.pack(provides_time);
    bs.pack(provides_geometry);
    bs.pack(variables);
    bs.pack(x_axis_variable);
    bs.pack(y_axis_variable);
    bs.pack(z_axis_variable);
    bs.pack(t_axis_variable);
    bs.pack(periodic_in_x);
    bs.pack(calendar);
    bs.pack(t_units);
    bs.pack(filename_time_template);
    bs.pack(clamp_dimensions_of_one);
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader_internals::cf_reader_options::from_stream(
    teca_binary_stream &bs)
{
    bs.unpack(name);
    bs.unpack(regex);
    bs.unpack(provides_time);
    bs.unpack(provides_geometry);
    bs.unpack(variables);
    bs.unpack(x_axis_variable);
    bs.unpack(y_axis_variable);
    bs.unpack(z_axis_variable);
    bs.unpack(t_axis_variable);
    bs.unpack(periodic_in_x);
    bs.unpack(calendar);
    bs.unpack(t_units);
    bs.unpack(filename_time_template);
    bs.unpack(clamp_dimensions_of_one);
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader_internals::cf_reader_options::parse_line(
    char *line, unsigned long line_no)
{
    if (strncmp("name", line, 5) == 0)
    {
        if (!name.empty())
        {
            TECA_ERROR("Duplicate name label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, name))
        {
            TECA_ERROR("Syntax error when parsing name on line " << line_no)
            return -1;
        }
    }
    else if (strncmp("regex", line, 5) == 0)
    {
        if (!regex.empty())
        {
            TECA_ERROR("Duplicate regex label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, regex))
        {
            TECA_ERROR("Syntax error when parsing regex on line " << line_no)
            return -1;
        }
    }
    else if (strncmp("variables", line, 9) == 0)
    {
        if (!variables.empty())
        {
            TECA_ERROR("Duplicate regex label found on line " << line_no)
            return -1;
        }

        std::vector<char*> tmp;
        if (teca_string_util::tokenize(line, '=', tmp) || (tmp.size() != 2))
        {
            TECA_ERROR("Invalid variables specifier : \"" << line
                << "\" on line " << line_no)
            return -1;
        }

        std::vector<char*> vars;
        if (teca_string_util::tokenize(tmp[1], ',', vars) || (vars.size() < 1))
        {
            TECA_ERROR("Invalid variables specifier : \"" << line
                << "\" on line " << line_no)
            return -1;
        }

        size_t n_vars  = vars.size();
        for (size_t i = 0; i < n_vars; ++i)
        {
            char *v = vars[i];
            if (teca_string_util::skip_pad(v))
            {
                TECA_ERROR("Invalid variable name on line " << line_no)
                return -1;
            }
            variables.push_back(v);
        }
    }
    else if (strncmp("provides_time", line, 11) == 0)
    {
        if (provides_time)
        {
            TECA_ERROR("Duplicate provides_time label found on line " << line_no)
            return -1;
        }

        provides_time = 1;
    }
    else if (strncmp("provides_geometry", line, 15) == 0)
    {
        if (provides_geometry)
        {
            TECA_ERROR("Duplicate provides_geometry label found on line " << line_no)
            return -1;
        }

        provides_geometry = 1;
    }
    else if (strncmp("x_axis_variable", line, 15) == 0)
    {
        if (!x_axis_variable.empty())
        {
            TECA_ERROR("Duplicate x_axis_variable label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, x_axis_variable))
        {
            TECA_ERROR("Syntax error when parsing x_axis_variable on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("y_axis_variable", line, 15) == 0)
    {
        if (!y_axis_variable.empty())
        {
            TECA_ERROR("Duplicate y_axis_variable label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, y_axis_variable))
        {
            TECA_ERROR("Syntax error when parsing y_axis_variable on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("z_axis_variable", line, 15) == 0)
    {
        if (!z_axis_variable.empty())
        {
            TECA_ERROR("Duplicate z_axis_variable label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, z_axis_variable))
        {
            TECA_ERROR("Syntax error when parsing z_axis_variable on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("periodic_in_x", line, 15) == 0)
    {
        if (!(periodic_in_x < 0))
        {
            TECA_ERROR("Duplicate periodic_in_x label found on line " << line_no)
            return -1;
        }

        std::string tmp;
        bool val = false;
        if (teca_string_util::extract_value<std::string>(line, tmp)
            || teca_string_util::string_tt<bool>::convert(tmp.c_str(), val))
        {
            TECA_ERROR("Syntax error when parsing periodic_in_x on line " << line_no)
            return -1;
        }

        periodic_in_x = val ? 1 : 0;

        return 1;
    }
    else if (strncmp("t_axis_variable", line, 15) == 0)
    {
        if (!t_axis_variable.empty())
        {
            TECA_ERROR("Duplicate t_axis_variable label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, t_axis_variable))
        {
            TECA_ERROR("Syntax error when parsing t_axis_variable on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("calendar", line, 8) == 0)
    {
        if (!calendar.empty())
        {
            TECA_ERROR("Duplicate calendar label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, calendar))
        {
            TECA_ERROR("Syntax error when parsing calendar on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("t_units", line, 15) == 0)
    {
        if (!t_units.empty())
        {
            TECA_ERROR("Duplicate t_units label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, t_units))
        {
            TECA_ERROR("Syntax error when parsing t_units on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("filename_time_template", line, 15) == 0)
    {
        if (!filename_time_template.empty())
        {
            TECA_ERROR("Duplicate filename_time_template label found on line " << line_no)
            return -1;
        }

        if (teca_string_util::extract_value<std::string>(line, filename_time_template))
        {
            TECA_ERROR("Syntax error when parsing filename_time_template on line " << line_no)
            return -1;
        }

        return 1;
    }
    else if (strncmp("clamp_dimensions_of_one", line, 23) == 0)
    {
        if (!(clamp_dimensions_of_one < 0))
        {
            TECA_ERROR("Duplicate clamp_dimensions_of_one label found on line " << line_no)
            return -1;
        }

        std::string tmp;
        bool val = false;
        if (teca_string_util::extract_value<std::string>(line, tmp)
            || teca_string_util::string_tt<bool>::convert(tmp.c_str(), val))
        {
            TECA_ERROR("Syntax error when parsing clamp_dimensions_of_one on line " << line_no)
            return -1;
        }

        clamp_dimensions_of_one = val ? 1 : 0;

        return 1;
    }

    return 0;
}





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
    teca_file_util::line_buffer &lines,
    teca_multi_cf_reader_internals::cf_reader_options &opts)
{
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
        if (opts.parse_line(l, lno) < 0)
            return -1;
    }

    // the section was valid if at least regex and vars were found
    bool have_regex = !opts.regex.empty();
    int have_variables = !opts.variables.empty();

    return (have_regex && have_variables) ? 0 : -1;
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader_internals::add_reader_instance(
    const cf_reader_options &options)
{
    // construct and intialize the reader
    p_teca_cf_reader reader = teca_cf_reader::New();

    if (options.name.empty())
    {
        TECA_ERROR("Invalid reader key. It must not be empty")
        return -1;
    }

    this->readers[options.name] =
        p_cf_reader_instance(new cf_reader_instance(reader,
                std::set<std::string>(options.variables.begin(),
                    options.variables.end()), options));

    if (options.provides_time)
        this->time_reader = options.name;

    if (options.provides_geometry)
        this->geometry_reader = options.name;

    return 0;
}






// --------------------------------------------------------------------------
teca_multi_cf_reader::teca_multi_cf_reader() :
    input_file(""),
    x_axis_variable("lon"),
    y_axis_variable("lat"),
    z_axis_variable(""),
    t_axis_variable("time"),
    calendar(""),
    t_units(""),
    filename_time_template(""),
    periodic_in_x(0),
    max_metadata_ranks(-1),
    clamp_dimensions_of_one(0),
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
            "a file dedscribing the dataset layout")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "name of variable that has x axis coordinates")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "name of variable that has y axis coordinates")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "name of variable that has z axis coordinates")
        TECA_POPTS_GET(std::string, prefix, t_axis_variable,
            "name of variable that has t axis coordinates")
        TECA_POPTS_GET(std::string, prefix, calendar,
            "name of variable that has the time calendar")
        TECA_POPTS_GET(std::string, prefix, t_units,
            "a std::get_time template for decoding time from the input filename")
        TECA_POPTS_GET(std::string, prefix, filename_time_template,
            "name of variable that has the time unit")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, t_values,
            "name of variable that has t axis values set by the"
            "the user if the file doesn't have time variable set")
        TECA_POPTS_GET(int, prefix, periodic_in_x,
            "the dataset has a periodic boundary in the x direction")
        TECA_POPTS_GET(int, prefix, max_metadata_ranks,
            "set the max number of ranks for reading metadata")
        TECA_POPTS_GET(int, prefix, clamp_dimensions_of_one,
            "If set clamp requested axis extent in where the request is out of"
            " bounds and the coordinate array dimension is 1.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, t_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, calendar)
    TECA_POPTS_SET(opts, std::string, prefix, t_units)
    TECA_POPTS_SET(opts, std::string, prefix, filename_time_template)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, t_values)
    TECA_POPTS_SET(opts, int, prefix, periodic_in_x)
    TECA_POPTS_SET(opts, int, prefix, max_metadata_ranks)
    TECA_POPTS_SET(opts, int, prefix, clamp_dimensions_of_one)
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

    int num_readers = 0;
    teca_binary_stream bs;

    using reader_option_t = teca_multi_cf_reader_internals::cf_reader_options;

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
        reader_option_t g_opts;

        std::vector<reader_option_t> section_opts;

        while (lines)
        {
            int ierr = 0;
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
            else if ((ierr = g_opts.parse_line(l, lno)))
            {
                // report parsing error
                if (ierr < 0)
                    return -1;
            }
            else if (strcmp("[cf_reader]", l) == 0)
            {
                reader_option_t opts;
                if (teca_multi_cf_reader_internals::parse_cf_reader_section(lines, opts))
                {
                    TECA_ERROR("Failed to parse [cf_reader] section on line " << lno)
                    return -1;
                }

                // always give a name
                if (opts.name.empty())
                    opts.name = std::to_string(section_opts.size());

                // look for and replace %data_root% if it is present
                if (!g_data_root.empty())
                {
                    size_t loc = opts.regex.find("%data_root%");
                    if (loc != std::string::npos)
                        opts.regex.replace(loc, 11, g_data_root);
                }

                if (!g_regex.empty())
                {
                    size_t loc = opts.regex.find("%regex%");
                    if (loc != std::string::npos)
                        opts.regex.replace(loc, 7, g_regex);
                }

                // save
                section_opts.push_back(opts);
            }
        }

        // serialize
        g_opts.to_stream(bs);

        num_readers = section_opts.size();
        bs.pack(num_readers);

        for (int i = 0; i < num_readers; ++i)
            section_opts[i].to_stream(bs);
    }

    // share
    bs.broadcast(this->get_communicator());

    // deserialize
    this->internals->global_options.from_stream(bs);

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
        teca_multi_cf_reader_internals::cf_reader_options options;

        options.from_stream(bs);

        num_time_readers += options.provides_time;
        num_geometry_readers += options.provides_geometry;

        this->internals->add_reader_instance(options);
    }

    if (num_time_readers != 1)
    {
        TECA_ERROR(<< num_time_readers << " readers provide time."
            " One and only one reader can provide time.")
        return -1;
    }

    if (num_geometry_readers != 1)
    {
        TECA_ERROR(<< num_geometry_readers << " readers provide geometry."
            " One and only one reader can provide mesh geometry.")
        return -1;
    }

    this->input_file = input_file;

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
void teca_multi_cf_reader::set_x_axis_variable(const std::string &var)
{
    if (this->x_axis_variable != var)
    {
        this->x_axis_variable = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_x_axis_variable() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->x_axis_variable;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_x_axis_variable(
        this->internals->global_options.get_x_axis_variable(
            this->x_axis_variable));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_y_axis_variable(const std::string &var)
{
    if (this->y_axis_variable != var)
    {
        this->y_axis_variable = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_y_axis_variable() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->y_axis_variable;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_y_axis_variable(
        this->internals->global_options.get_y_axis_variable(
            this->y_axis_variable));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_z_axis_variable(const std::string &var)
{
    if (this->z_axis_variable != var)
    {
        this->z_axis_variable = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_z_axis_variable() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->z_axis_variable;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_z_axis_variable(
        this->internals->global_options.get_z_axis_variable(
            this->z_axis_variable));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_periodic_in_x(int var)
{
    if (this->periodic_in_x != var)
    {
        this->periodic_in_x = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::get_periodic_in_x() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->periodic_in_x;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return -1;
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_periodic_in_x(
        this->internals->global_options.get_periodic_in_x(
            this->periodic_in_x));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_t_axis_variable(const std::string &var)
{
    if (this->t_axis_variable != var)
    {
        this->t_axis_variable = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_t_axis_variable() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->t_axis_variable;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_t_axis_variable(
        this->internals->global_options.get_t_axis_variable(
            this->t_axis_variable));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_calendar(const std::string &var)
{
    if (this->calendar != var)
    {
        this->calendar = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_calendar() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->time_reader.empty())
    {
        // the time reader wasn't established yet, fall back to
        // the current property value
        return this->calendar;
    }

    // get the time reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->time_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->time_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_calendar(
        this->internals->global_options.get_calendar(
            this->calendar));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_t_units(const std::string &var)
{
    if (this->t_units != var)
    {
        this->t_units = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_t_units() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->time_reader.empty())
    {
        // the time reader wasn't established yet, fall back to
        // the current property value
        return this->t_units;
    }

    // get the time reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->time_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->time_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_t_units(
        this->internals->global_options.get_t_units(
            this->t_units));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_filename_time_template(const std::string &var)
{
    if (this->filename_time_template != var)
    {
        this->filename_time_template = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
std::string teca_multi_cf_reader::get_filename_time_template() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->time_reader.empty())
    {
        // the time reader wasn't established yet, fall back to
        // the current property value
        return this->filename_time_template;
    }

    // get the time reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->time_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->time_reader << "\" found")
        return "";
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_filename_time_template(
        this->internals->global_options.get_filename_time_template(
            this->filename_time_template));
}

// --------------------------------------------------------------------------
void teca_multi_cf_reader::set_clamp_dimensions_of_one(int var)
{
    if (this->clamp_dimensions_of_one != var)
    {
        this->clamp_dimensions_of_one = var;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::get_clamp_dimensions_of_one() const
{
    // settings from the MCF file should override algorithm properties
    // however, this may be called any time before or after the readers
    // are set up.

    if (this->internals->geometry_reader.empty())
    {
        // the geometry reader wasn't established yet, fall back to
        // the current property value
        return this->clamp_dimensions_of_one;
    }

    // get the geometry reader instance
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(this->internals->geometry_reader);

    if (it == this->internals->readers.end())
    {
        TECA_ERROR("No reader named \""
            << this->internals->geometry_reader << "\" found")
        return -1;
    }

    // values in the configuration file take precedence over the member variable
    // with in the configuration file, section options take precedence over
    // globally scoped options
    return it->second->options.get_clamp_dimensions_of_one(
        this->internals->global_options.get_clamp_dimensions_of_one(
            this->clamp_dimensions_of_one));
}

// --------------------------------------------------------------------------
int teca_multi_cf_reader::add_reader(const std::string &regex,
    const std::string &key, int provides_time, int provides_geometry,
    const std::vector<std::string> &variables)
{
    teca_multi_cf_reader_internals::cf_reader_options options;

    options.name = key;
    options.regex = regex;
    options.provides_time = provides_time;
    options.provides_geometry = provides_geometry;
    options.variables = variables;

    return this->internals->add_reader_instance(options);
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
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(key);

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
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(key);

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
    teca_multi_cf_reader_internals::reader_map_t::iterator it =
        this->internals->readers.find(key);

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
    const teca_multi_cf_reader_internals::cf_reader_options &global_options = this->internals->global_options;

    teca_multi_cf_reader_internals::reader_map_t::iterator it = this->internals->readers.begin();
    teca_multi_cf_reader_internals::reader_map_t::iterator end = this->internals->readers.end();
    for (; it != end; ++it)
    {
        const std::string &key = it->first;
        teca_multi_cf_reader_internals::p_cf_reader_instance &inst = it->second;

        // configure the reader. settings from the MCF file override the
        // algorithm properties. with in the MCF file settings from specific
        // reader section override global settings.
        inst->reader->set_files_regex(inst->options.regex);

        inst->reader->set_x_axis_variable(
            inst->options.get_x_axis_variable(
                global_options.get_x_axis_variable(
                    this->x_axis_variable)));

        inst->reader->set_y_axis_variable(
            inst->options.get_y_axis_variable(
                global_options.get_y_axis_variable(
                    this->y_axis_variable)));

        inst->reader->set_z_axis_variable(
            inst->options.get_z_axis_variable(
                global_options.get_z_axis_variable(
                    this->z_axis_variable)));

        inst->reader->set_t_axis_variable(
            inst->options.get_t_axis_variable(
                global_options.get_t_axis_variable(
                    this->t_axis_variable)));

        inst->reader->set_periodic_in_x(
            inst->options.get_periodic_in_x(
                global_options.get_periodic_in_x(
                    this->periodic_in_x)));

        inst->reader->set_calendar(
            inst->options.get_calendar(
                global_options.get_calendar(
                    this->calendar)));

        inst->reader->set_t_units(
            inst->options.get_t_units(
                global_options.get_t_units(
                    this->t_units)));

        inst->reader->set_filename_time_template(
            inst->options.get_filename_time_template(
                global_options.get_filename_time_template(
                    this->filename_time_template)));

        inst->reader->set_clamp_dimensions_of_one(
            inst->options.get_clamp_dimensions_of_one(
                global_options.get_clamp_dimensions_of_one(
                    this->clamp_dimensions_of_one)));

        if (!this->t_values.empty())
            inst->reader->set_t_values(this->t_values);

        if (this->max_metadata_ranks >= 0)
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
                TECA_ERROR("Failed to get the x-axis varaible name")
                return teca_metadata();
            }
            coords_out.set("x_variable", x_variable);

            if (!inst->reader->get_x_axis_variable().empty())
            {
                teca_metadata x_atts;
                if (atts_in.get(x_variable, x_atts))
                {
                    TECA_ERROR("Failed to get attributes for the x-axis variable \""
                        << x_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(x_variable, x_atts);
            }

            p_teca_variant_array x = coords_in.get("x");
            if (!x)
            {
                TECA_ERROR("Failed to get the x-axis")
                return teca_metadata();
            }
            coords_out.set("x", x);

            // pass x axis and attributes
            std::string y_variable;
            if (coords_in.get("y_variable", y_variable))
            {
                TECA_ERROR("Failed to get the y-axis varaible name")
                return teca_metadata();
            }
            coords_out.set("y_variable", y_variable);

            if (!inst->reader->get_y_axis_variable().empty())
            {
                teca_metadata y_atts;
                if (atts_in.get(y_variable, y_atts))
                {
                    TECA_ERROR("Failed to get attributes for the y-axis variable \""
                        << y_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(y_variable, y_atts);
            }

            p_teca_variant_array y = coords_in.get("y");
            if (!y)
            {
                TECA_ERROR("Failed to get the y-axis")
                return teca_metadata();
            }
            coords_out.set("y", y);

            // pass y axis and attributes
            std::string z_variable;
            if (coords_in.get("z_variable", z_variable))
            {
                TECA_ERROR("Failed to get the z-axis varaible name")
                return teca_metadata();
            }
            coords_out.set("z_variable", z_variable);

            if (!inst->reader->get_z_axis_variable().empty())
            {
                teca_metadata z_atts;
                if (atts_in.get(z_variable, z_atts))
                {
                    TECA_ERROR("Failed to get attributes for the z-axis variable \""
                        << z_variable << "\"")
                    return teca_metadata();
                }
                atts_out.set(z_variable, z_atts);
            }

            p_teca_variant_array z = coords_in.get("z");
            if (!z)
            {
                TECA_ERROR("Failed to get the z-axis")
                return teca_metadata();
            }
            coords_out.set("z", z);

            // pass periodicity
            int periodic = 0;
            coords_in.get("periodic_in_x", periodic);
            coords_out.set("periodic_in_x", periodic);

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

        // make a pass over the arrays. if this reader provides the array
        // add it to the associated router and don't include it in later
        // searches.
        std::vector<std::string> arrays_left;

        size_t n_arrays = req_arrays.size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            const std::string &array = req_arrays[i];
            std::set<std::string>::iterator vit = inst->variables.find(array);

            if (vit == inst->variables.end())
                arrays_left.push_back(array);
            else
                array_router[key].push_back(array);
        }

        req_arrays.swap(arrays_left);
    }

    // none of the readers could provide the remaining arrays.
    if (!req_arrays.empty())
    {
        TECA_ERROR("No reader provides the requested arrays " << req_arrays)
        return nullptr;
    }

    // read mesh geometry, and the arrays provided by this reader first.
    // the returned mesh becomes the output.
    p_teca_cartesian_mesh mesh_out;

    const std::string &geom_reader = this->internals->geometry_reader;

    const std::vector<std::string> &geom_arrays =
        array_router[this->internals->geometry_reader];

    if (teca_multi_cf_reader_internals::read_arrays(
        this->internals->readers[geom_reader]->reader, request, geom_arrays,
        mesh_out))
    {
        TECA_ERROR("Geometry reader \"" << geom_reader
            << "\" failed to read arrays " << geom_arrays)
        return nullptr;
    }

    // get the output metadata and the array attributes
    teca_metadata &md_out = mesh_out->get_metadata();

    teca_metadata attributes;
    if (md_out.get("attributes", attributes))
    {
        TECA_ERROR("Geometry reader \"" << geom_reader
            << "\" failed to get attributes")
        return nullptr;
    }

    // read the rest of the arrays. iterate over each reader, get the list
    // of requested arrays that it sources, read, and transfer the arrays
    // and metadata to the output
    it = this->internals->readers.begin();
    for (; it != end; ++it)
    {
        const std::string &key = it->first;
        teca_multi_cf_reader_internals::p_cf_reader_instance &inst = it->second;

        // skip the geometry reader, we already read those above
        if (key == this->internals->geometry_reader)
            continue;

        // read the reader's arrays
        const std::vector<std::string> &arrays = array_router[key];
        size_t n_arrays = arrays.size();

        p_teca_cartesian_mesh tmp;
        if (teca_multi_cf_reader_internals::read_arrays(inst->reader,
            request, arrays, tmp))
        {
            TECA_ERROR("Reader \"" << key << "\" failed to read arrays " << arrays)
            return nullptr;
        }

        // transfer the arrays into the output
        mesh_out->get_point_arrays()->append(tmp->get_point_arrays());
        mesh_out->get_cell_arrays()->append(tmp->get_cell_arrays());
        mesh_out->get_x_edge_arrays()->append(tmp->get_x_edge_arrays());
        mesh_out->get_y_edge_arrays()->append(tmp->get_y_edge_arrays());
        mesh_out->get_z_edge_arrays()->append(tmp->get_z_edge_arrays());
        mesh_out->get_x_face_arrays()->append(tmp->get_x_face_arrays());
        mesh_out->get_y_face_arrays()->append(tmp->get_y_face_arrays());
        mesh_out->get_z_face_arrays()->append(tmp->get_z_face_arrays());
        mesh_out->get_information_arrays()->append(tmp->get_information_arrays());

        // transfer the array attributes
        teca_metadata &md_tmp = tmp->get_metadata();

        teca_metadata atrs;
        if (md_tmp.get("attributes", atrs))
        {
            TECA_ERROR("Reader \"" << key << " failed to get attributes")
            return nullptr;
        }

        for (size_t i = 0; i < n_arrays; ++i)
        {
            const std::string &array_name = arrays[i];
            teca_metadata array_atts;
            if (atrs.get(array_name, array_atts))
            {
                TECA_ERROR("Reader \"" << key
                    << "\" failed to get attributes for array \""
                    << array_name << "\"")
                atrs.to_stream(std::cerr);
                return nullptr;
            }

            attributes.set(array_name, array_atts);
        }
    }

    // update the array attribute metadata
    md_out.set("attributes", attributes);

    return mesh_out;
}
