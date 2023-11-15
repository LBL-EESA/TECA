#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_netcdf_util.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_coordinate_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_app_util.h"
#include "teca_array_attributes.h"

#if defined(TECA_HAS_UDUNITS)
#include "teca_calcalcs.h"

#include "teca_calendar_util.h"

using teca_calendar_util::day_iterator;
using teca_calendar_util::month_iterator;
using teca_calendar_util::season_iterator;
using teca_calendar_util::year_iterator;
#endif

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <boost/program_options.hpp>

using namespace std;
using boost::program_options::value;
using namespace teca_variant_array_util;

#if defined(TECA_HAS_UDUNITS)
// --------------------------------------------------------------------------
template <typename it_t>
long count(const const_p_teca_variant_array &time,
    const std::string &calendar, const std::string &units, long i0, long i1)
{
    it_t it;

    if (it.initialize(time, units, calendar, i0, i1))
        return -1;

    long n_int = 0;
    while (it)
    {
        teca_calendar_util::time_point p0, p1;
        it.get_next_interval(p0, p1);
        ++n_int;
    }

    return n_int;
}
#endif

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // initialize comand line options description set up some comon options to
    // simplify use for most comon scenarios
    int help_width = 100;
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most comonly used. Information\n"
        "on all available options can be displayed using --advanced_help\n\n"
        "Basic comand line options", help_width, help_width - 4
        );
    basic_opt_defs.add_options()

        ("input_file", value<std::string>(), "\na teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\na teca_cf_reader regex identyifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nname of x coordinate variable\n")
        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nname of y coordinate variable\n")
        ("z_axis_variable", value<std::string>()->default_value(""),
            "\nname of z coordinate variable. When processing 3D set this to"
            " the variable containing vertical coordinates. When empty the"
            " data will be treated as 2D.\n")

        ("start_date", value<std::string>(), "\nThe first time to process in 'Y-M-D h:m:s'"
            " format. Note: There must be a space between the date and time specification\n")
        ("end_date", value<std::string>(), "\nThe last time to process in 'Y-M-D h:m:s' format\n")

        ("help", "\ndisplays documentation for application specific command line options\n")
        ("advanced_help", "\ndisplays documentation for algorithm specific command line options\n")
        ("full_help", "\ndisplays both basic and advanced documentation together\n")
        ;

    // add all options from each pipeline stage for more advanced use
    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "Advanced comand line options", help_width, help_width - 4
        );

    // create the pipeline stages here, they contain the
    // documentation and parse comand line.
    // objects report all of their properties directly
    // set default options here so that comand line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates norm_coords = teca_normalize_coordinates::New();
    norm_coords->get_properties_description("norm_coords", advanced_opt_defs);

    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    int ierr = 0;
    variables_map opt_vals;
    if ((ierr = teca_app_util::process_command_line_help(
        mpi_man.get_comm_rank(), argc, argv, basic_opt_defs,
        advanced_opt_defs, all_opt_defs, opt_vals)))
    {
        if (ierr == 1)
            return 0;
        return -1;
    }

    // pass comand line arguments into the pipeline objects
    // advanced options are procesed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    norm_coords->set_properties("norm_coords", opt_vals);

    // now pas in the basic options, these are procesed
    // last so that they will take precedence
    if (!opt_vals["x_axis_variable"].defaulted())
    {
        cf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<std::string>());
        mcf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<std::string>());
    }

    if (!opt_vals["y_axis_variable"].defaulted())
    {
        cf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<std::string>());
        mcf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<std::string>());
    }

    if (!opt_vals["z_axis_variable"].defaulted())
    {
        cf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
        mcf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
    }

    std::string x_var;
    std::string y_var;
    std::string z_var;

    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");

    // validate the input method
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (CMIP6 datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    p_teca_algorithm reader;
    if (opt_vals.count("input_file"))
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        x_var = mcf_reader->get_x_axis_variable();
        y_var = mcf_reader->get_y_axis_variable();
        z_var = mcf_reader->get_z_axis_variable();
        reader = mcf_reader;
    }
    else if (opt_vals.count("input_regex"))
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        x_var = cf_reader->get_x_axis_variable();
        y_var = cf_reader->get_y_axis_variable();
        z_var = cf_reader->get_z_axis_variable();
        reader = cf_reader;
    }
    norm_coords->set_input_connection(reader->get_output_port());
    norm_coords->set_verbose(2);

    std::string time_i;
    if (opt_vals.count("start_date"))
        time_i = opt_vals["start_date"].as<string>();

    std::string time_j;
    if (opt_vals.count("end_date"))
        time_j = opt_vals["end_date"].as<string>();

    // run the reporting phase of the pipeline
    teca_metadata md = norm_coords->update_metadata();

    // from here on out just rank 0
    if (rank == 0)
    {
        //md.to_stream(cerr);

        // extract metadata
        int n_files = -1;
        if (md.has("files"))
        {
            n_files = md.get("files")->size();
        }

        teca_metadata atrs;
        if (md.get("attributes", atrs))
        {
            TECA_FATAL_ERROR("metadata mising attributes")
            return -1;
        }

        teca_metadata time_atts;
        std::string calendar;
        std::string units;
        if (atrs.get("time", time_atts)
           || time_atts.get("calendar", calendar)
           || time_atts.get("units", units))
        {
            TECA_FATAL_ERROR("failed to determine the calendaring parameters")
            return -1;
        }

        teca_metadata coords;
        p_teca_variant_array t;
        if (md.get("coordinates", coords) || !(t = coords.get("t")))
        {
            TECA_FATAL_ERROR("failed to determine time coordinate")
            return -1;
        }

        p_teca_double_array time =
            std::dynamic_pointer_cast<teca_double_array>(t);

        if (!time)
        {
            // convert to double precision
            size_t n = t->size();
            double *p_time = nullptr;
            std::tie(time, p_time) = ::New<teca_double_array>(n);
            VARIANT_ARRAY_DISPATCH(t.get(),
                auto [sp_t, p_t] = get_host_accessible<CTT>(t);
                for (size_t i = 0; i < n; ++i)
                    p_time[i] = static_cast<double>(p_t[i]);
                )
        }

        unsigned long n_time_steps = 0;
        if (md.get("number_of_time_steps", n_time_steps))
        {
            TECA_FATAL_ERROR("failed to deermine the number of steps")
            return -1;
        }

        unsigned long i0 = 0;
        unsigned long i1 = n_time_steps - 1;

        // human readable first time available
        int Y = 0, M = 0, D = 0, h = 0, m = 0;
        double s = 0;
#if defined(TECA_HAS_UDUNITS)
        if (teca_calcalcs::date(time->get(i0), &Y, &M, &D, &h, &m, &s,
            units.c_str(), calendar.c_str()))
        {
            TECA_FATAL_ERROR("failed to detmine the first available time in the file")
            return -1;
        }
#else
        TECA_FATAL_ERROR("UDUnits is required for human readable dates")
#endif
        std::ostringstream oss;
        oss << Y << "-" << M << "-" << D << " " << h << ":" << m << ":" << s;
        std::string time_0(oss.str());

        // human readbale last time available
        Y = 0, M = 0, D = 0, h = 0, m = 0, s = 0;
#if defined(TECA_HAS_UDUNITS)
        if (teca_calcalcs::date(time->get(i1), &Y, &M, &D, &h, &m, &s,
            units.c_str(), calendar.c_str()))
        {
            TECA_FATAL_ERROR("failed to detmine the last available time in the file")
            return -1;
        }
#endif
        oss.str("");
        oss << Y << "-" << M << "-" << D << " " << h << ":" << m << ":" << s;
        std::string time_n(oss.str());

        // look for requested time step range, start
        if (!time_i.empty())
        {
            if (teca_coordinate_util::time_step_of(
                 time, true, true, calendar, units, time_i, i0))
            {
                TECA_FATAL_ERROR("Failed to locate time step for start date \""
                    << time_i << "\"")
                return -1;
            }
        }

        // end step
        if (!time_j.empty())
        {
            if (teca_coordinate_util::time_step_of(
                 time, false, true, calendar, units, time_j, i1))
            {
                TECA_FATAL_ERROR("Failed to locate time step for end date \""
                    << time_j << "\"")
                return -1;
            }
        }

        oss.str("");
        oss << "A total of " <<  n_time_steps << " steps available";
        if (n_files > 0)
            oss << " in " << n_files << " files";
        oss << ". Using the " << calendar << " calendar. Times are specified in units of "
            << units << ". The available times range from " << time_0 << " (" << time->get(0)
            << ") to " << time_n << " (" << time->get(time->size()-1) << ").";

        if (!time_i.empty() || !time_j.empty())
        {
            oss << " The requested range contains " << i1 - i0 + 1 << " time steps and ranges from "
                << time_i << " (" << time->get(i0) << ") to " << time_j << " (" << time->get(i1) << ") "
                << "), starts at time step " << i0 << " and goes to time step " << i1 << ".";
        }

#if defined(TECA_HAS_UDUNITS)
        oss << " The available data contains:";
        if (long ny = count<year_iterator>(time, calendar, units, i0, i1))
            oss << " " << ny << " years;";

        if (long ns = count<season_iterator>(time, calendar, units, i0, i1))
            oss << " " << ns << " seasons;";

        if (long nm = count<month_iterator>(time, calendar, units, i0, i1))
            oss << " " << nm << " months;";

        if (long nd = count<day_iterator>(time, calendar, units, i0, i1))
            oss << " " << nd << " days;";
#endif

        std::string report(oss.str());

        std::string::iterator it = report.begin();
        std::string::iterator end = report.end();
        unsigned long i = 0;
        unsigned long line_len = 74;
        std::cerr << std::endl;
        for (; it != end; ++it, ++i)
        {
            if ((i >= line_len) && (*it == ' '))
            {
                std::cerr << std::endl;
                ++it;
                i = 0;
            }
            std::cerr << *it;
        }
        std::cerr << std::endl << std::endl;

        // report the mesh dimensionality and coordinates
        std::cerr << "Mesh dimension: " << (z_var.empty() ? 2 : 3)
            << "D" << std::endl;

        std::cerr << "Mesh coordinates: " << x_var << ", " << y_var;
        if (!z_var.empty())
        {
            std::cerr << ", " << z_var;
        }
        std::cerr << std::endl;

        // report extents
        long extent[6] = {0l};
        if (!md.get("whole_extent", extent, 6))
        {
            std::cerr << "Mesh extents: " << extent[0] << ", " << extent[1]
                << ", " << extent[2] << ", " << extent[3];
            if (!z_var.empty())
            {
                std::cerr << ", " << extent[4] << ", " << extent[5];
            }
            std::cerr << std::endl;
        }

        // report bounds
        double bounds[6] = {0.0};
        if (!md.get("bounds", bounds, 6))
        {
            std::cerr << "Mesh bounds: " << bounds[0] << ", " << bounds[1]
                << ", " << bounds[2] << ", " << bounds[3];
            if (!z_var.empty())
            {
                std::cerr << ", " << bounds[4] << ", " << bounds[5];
            }
            std::cerr << std::endl;
        }


        // report the arrays
        size_t n_arrays = atrs.size();

        // column widths
        int aiw = 4;
        int anw = 8;
        int atw = 8;
        int acw = 14;
        int adw = 15;
        int asw = 9;

        // column data
        std::vector<std::string> ai;
        std::vector<std::string> an;
        std::vector<std::string> at;
        std::vector<std::string> ac;
        std::vector<std::string> ad;
        std::vector<std::string> as;

        ai.reserve(n_arrays);
        an.reserve(n_arrays);
        at.reserve(n_arrays);
        ac.reserve(n_arrays);
        ad.reserve(n_arrays);
        as.reserve(n_arrays);

        for (size_t i = 0; i < n_arrays; ++i)
        {
            std::string array;
            atrs.get_name(i, array);

            // get metadata
            teca_metadata atts;
            int type = 0;
            int id = 0;
            p_teca_size_t_array dims;
            p_teca_string_array dim_names;
            int centering = 0;
            int n_active_dims = 0;

            if (atrs.get(array, atts)
                || atts.get("cf_type_code", 0, type)
                || atts.get("cf_id", 0, id)
                || atts.get("centering", centering)
                || atts.get("n_active_dims", n_active_dims)
                || !(dims = std::dynamic_pointer_cast<teca_size_t_array>(atts.get("cf_dims")))
                || !(dim_names = std::dynamic_pointer_cast<teca_string_array>(atts.get("cf_dim_names"))))
            {
                // TODO -- Michael's CAM5 sometimes triggers this with an empty array name
                //TECA_FATAL_ERROR("metadata issue in array " << i << "\"" << array << "\"")
                continue;
            }

            // id
            ai.push_back(std::to_string(i+1));
            aiw = std::max<int>(aiw, ai.back().size() + 4);

            // name
            an.push_back(array);
            anw = std::max<int>(anw, an.back().size() + 4);

            // type
            NC_DISPATCH(type,
                at.push_back(teca_netcdf_util::netcdf_tt<NC_NT>::name());
                )
            atw = std::max<int>(atw, at.back().size() + 4);

            // centering
            ac.push_back(teca_array_attributes::centering_to_string(centering) +
                std::string(" ") + std::to_string(n_active_dims) + "D");
            acw = std::max<int>(acw, ac.back().size() + 4);

            // dims
            int n_dims = dim_names->size();

            oss.str("");
            oss << "[" << dim_names->get(0);
            for (int i = 1; i < n_dims; ++i)
            {
                oss << ", " << dim_names->get(i);
            }
            oss << "]";
            ad.push_back(oss.str());
            adw = std::max<int>(adw, ad.back().size() + 4);

            // shape
            oss.str("");
            if (dim_names->get(0) == "time")
                oss << "[" << n_time_steps;
            else
               oss << "[" << dims->get(0);
            for (int i = 1; i < n_dims; ++i)
            {
                if (dim_names->get(i) == "time")
                    oss << ", " << n_time_steps;
                else
                    oss << ", " << dims->get(i);
            }
            oss << "]";
            as.push_back(oss.str());
            asw = std::max<int>(asw, as.back().size() + 4);
        }

        // update with the number found
        n_arrays = ai.size();

        std::cerr << std::endl
            << n_arrays << " data arrays available" << std::endl << std::endl
            << "  "
            << std::setw(aiw) << std::left << "Id"
            << std::setw(anw) << std::left << "Name"
            << std::setw(atw) << std::left << "Type"
            << std::setw(acw) << std::left << "Centering"
            << std::setw(adw) << std::left << "Dimensions"
            << std::setw(asw) << std::left << "Shape" << std::endl;

        int tw =  aiw + anw + atw + adw + acw + asw;
        for (int i = 0; i < tw; ++i)
            std::cerr << '-';
        std::cerr << std::endl;

        for (size_t i = 0; i < n_arrays; ++i)
        {
            std::cerr
                << "  "
                << std::setw(aiw) << std::left << ai[i]
                << std::setw(anw) << std::left << an[i]
                << std::setw(atw) << std::left << at[i]
                << std::setw(acw) << std::left << ac[i]
                << std::setw(adw) << std::left << ad[i]
                << std::setw(asw) << std::left << as[i]
                << std::endl;
        }

        std::cerr << std::endl;
    }

    return 0;
}
