#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_netcdf_util.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_coordinate_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
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

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // initialize comand line options description
    // set up some comon options to simplify use for most
    // comon scenarios
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most comonly used. Information\n"
        "on advanced options can be displayed using --advanced_help\n\n"
        "Basic comand line options", 120, -1
        );
    basic_opt_defs.add_options()

        ("input_file", value<std::string>(), "multi_cf_reader configuration file identifying"
            " simulation files tracks were generated from. when present data is read using"
            " the multi_cf_reader. use one of either --input_file or --input_regex.")

        ("input_regex", value<std::string>(), "cf_reader regex identyifying simulation files"
            " tracks were generated from. when present data is read using the cf_reader. use"
            " one of either --input_file or --input_regex.")

        ("x_axis", value<std::string>(), "name of x coordinate variable (lon)")
        ("y_axis", value<std::string>(), "name of y coordinate variable (lat)")
        ("z_axis", value<std::string>(), "name of z coordinate variable ()."
            " When processing 3D set this to the variable containing vertical coordinates."
            " When empty the data will be treated as 2D.")

        ("start_date", value<std::string>(), "first time to proces in Y-M-D h:m:s format")
        ("end_date", value<std::string>(), "first time to proces in Y-M-D h:m:s format")
        ("help", "display the basic options help")
        ("advanced_help", "display the advanced options help")
        ("full_help", "display all options help")
        ;

    // add all options from each pipeline stage for more advanced use
    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "Advanced comand line options", 120, -1
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

    // package basic and advanced options for display
    options_description all_opt_defs(-1, -1);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the comand line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(all_opt_defs).run(),
            opt_vals);

        if (opt_vals.count("help"))
        {
            if (rank == 0)
            {
                std::cerr << std::endl
                    << "usage: teca_metadata_probe [options]" << std::endl
                    << std::endl
                    << basic_opt_defs << std::endl
                    << std::endl;
            }
            return -1;
        }
        if (opt_vals.count("advanced_help"))
        {
            if (rank == 0)
            {
                std::cerr << std::endl
                    << "usage: teca_metadata_probe [options]" << std::endl
                    << std::endl
                    << advanced_opt_defs << std::endl
                    << std::endl;
            }
            return -1;
        }

        if (opt_vals.count("full_help"))
        {

            if (rank == 0)
            {
                std::cerr << std::endl
                    << "usage: teca_metadata_probe [options]" << std::endl
                    << std::endl
                    << all_opt_defs << std::endl
                    << std::endl;
            }
            return -1;
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        if (rank == 0)
        {
            TECA_ERROR("Error parsing comand line options. See --help "
                "for a list of supported options. " << e.what())
        }
        return -1;
    }

    // pass comand line arguments into the pipeline objects
    // advanced options are procesed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);

    // now pas in the basic options, these are procesed
    // last so that they will take precedence
    if (opt_vals.count("x_axis"))
    {
        cf_reader->set_x_axis_variable(opt_vals["x_axis"].as<std::string>());
        mcf_reader->set_x_axis_variable(opt_vals["x_axis"].as<std::string>());
    }

    if (opt_vals.count("y_axis"))
    {
        cf_reader->set_y_axis_variable(opt_vals["y_axis"].as<std::string>());
        mcf_reader->set_y_axis_variable(opt_vals["y_axis"].as<std::string>());
    }

    if (opt_vals.count("z_axis"))
    {
        cf_reader->set_z_axis_variable(opt_vals["z_axis"].as<std::string>());
        mcf_reader->set_z_axis_variable(opt_vals["z_axis"].as<std::string>());
    }

    std::string x_var;
    std::string y_var;
    std::string z_var;

    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");

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

    std::string time_i;
    if (opt_vals.count("start_date"))
        time_i = opt_vals["start_date"].as<string>();

    std::string time_j;
    if (opt_vals.count("end_date"))
        time_j = opt_vals["end_date"].as<string>();

    // some minimal check for mising options
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (rank == 0)
        {
            TECA_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    // run the reporting phase of the pipeline
    teca_metadata md = reader->update_metadata();

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
            TECA_ERROR("metadata mising attributes")
            return -1;
        }

        teca_metadata time_atts;
        std::string calendar;
        std::string units;
        if (atrs.get("time", time_atts)
           || time_atts.get("calendar", calendar)
           || time_atts.get("units", units))
        {
            TECA_ERROR("failed to determine the calendaring parameters")
            return -1;
        }

        teca_metadata coords;
        p_teca_variant_array t;
        if (md.get("coordinates", coords) || !(t = coords.get("t")))
        {
            TECA_ERROR("failed to determine time coordinate")
            return -1;
        }

        p_teca_double_array time =
            std::dynamic_pointer_cast<teca_double_array>(t);

        if (!time)
        {
            // convert to double precision
            size_t n = t->size();
            time = teca_double_array::New(n);
            double *p_time = time->get();
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                t.get(),
                NT *p_t = std::dynamic_pointer_cast<TT>(t)->get();
                for (size_t i = 0; i < n; ++i)
                    p_time[i] = static_cast<double>(p_t[i]);
                )
        }

        unsigned long n_time_steps = 0;
        if (md.get("number_of_time_steps", n_time_steps))
        {
            TECA_ERROR("failed to deermine the number of steps")
            return -1;
        }

        unsigned long i0 = 0;
        unsigned long i1 = n_time_steps - 1;

        // human readable first time available
        int Y = 0, M = 0, D = 0, h = 0, m = 0;
        double s = 0;
#if defined(TECA_HAS_UDUNITS)
        if (calcalcs::date(time->get(i0), &Y, &M, &D, &h, &m, &s,
            units.c_str(), calendar.c_str()))
        {
            TECA_ERROR("failed to detmine the first available time in the file")
            return -1;
        }
#else
        TECA_ERROR("UDUnits is required for human readable dates")
#endif
        std::ostringstream oss;
        oss << Y << "-" << M << "-" << D << " " << h << ":" << m << ":" << s;
        std::string time_0(oss.str());

        // human readbale last time available
        Y = 0, M = 0, D = 0, h = 0, m = 0, s = 0;
#if defined(TECA_HAS_UDUNITS)
        if (calcalcs::date(time->get(i1), &Y, &M, &D, &h, &m, &s,
            units.c_str(), calendar.c_str()))
        {
            TECA_ERROR("failed to detmine the last available time in the file")
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
                TECA_ERROR("Failed to locate time step for start date \""
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
                TECA_ERROR("Failed to locate time step for end date \""
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
                << "), starts at time step " << i0 << " and goes to time step " << i1;
        }

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

        // report the arrays
        size_t n_arrays = atrs.size();

        // column widths
        int aiw = 0;
        int anw = 0;
        int atw = 0;
        int adw = 0;
        int asw = 0;

        // column data
        std::vector<std::string> ai;
        std::vector<std::string> an;
        std::vector<std::string> at;
        std::vector<std::string> ad;
        std::vector<std::string> as;

        ai.reserve(n_arrays);
        an.reserve(n_arrays);
        at.reserve(n_arrays);
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

            if (atrs.get(array, atts)
                || atts.get("cf_type_code", 0, type)
                || atts.get("cf_id", 0, id)
                || !(dims = std::dynamic_pointer_cast<teca_size_t_array>(atts.get("cf_dims")))
                || !(dim_names = std::dynamic_pointer_cast<teca_string_array>(atts.get("cf_dim_names"))))
            {
                // TODO -- Michael's CAM5 sometimes triggers this with an empty array name
                //TECA_ERROR("metadata issue in array " << i << "\"" << array << "\"")
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
                at.push_back(teca_netcdf_util::netcdf_tt<NC_T>::name());
                )
            atw = std::max<int>(atw, at.back().size() + 4);

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
            << std::setw(adw) << std::left << "Dimensions"
            << std::setw(asw) << std::left << "Shape" << std::endl;

        int tw =  anw + atw + adw + asw;
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
                << std::setw(adw) << std::left << ad[i]
                << std::setw(asw) << std::left << as[i]
                << std::endl;
        }

        std::cerr << std::endl;
    }

    return 0;
}
