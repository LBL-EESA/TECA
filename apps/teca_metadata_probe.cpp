#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_cf_reader.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_coordinate_util.h"
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>

using namespace std;
using boost::program_options::value;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
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
        ("input_file", value<string>(), "file path to the simulation to search for tropical cyclones")
        ("input_regex", value<string>(), "regex matching simulation files to search for tropical cylones")
        ("start_date", value<string>(), "first time to proces in Y-M-D h:m:s format")
        ("end_date", value<string>(), "first time to proces in Y-M-D h:m:s format")
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
    p_teca_cf_reader sim_reader = teca_cf_reader::New();
    sim_reader->get_properties_description("sim_reader", advanced_opt_defs);

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
            cerr << endl
                << "usage: teca_data_probe [options]" << endl
                << endl
                << basic_opt_defs << endl
                << endl;
            return -1;
        }
        if (opt_vals.count("advanced_help"))
        {
            cerr << endl
                << "usage: teca_data_probe [options]" << endl
                << endl
                << advanced_opt_defs << endl
                << endl;
            return -1;
        }

        if (opt_vals.count("full_help"))
        {
            cerr << endl
                << "usage: teca_data_probe [options]" << endl
                << endl
                << all_opt_defs << endl
                << endl;
            return -1;
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        TECA_ERROR("Error parsing comand line options. See --help "
            "for a list of supported options. " << e.what())
        return -1;
    }

    // pass comand line arguments into the pipeline objects
    // advanced options are procesed first, so that the basic
    // options will override them
    sim_reader->set_properties("sim_reader", opt_vals);

    // now pas in the basic options, these are procesed
    // last so that they will take precedence
    if (opt_vals.count("input_file"))
        sim_reader->set_file_name(
            opt_vals["input_file"].as<string>());

    if (opt_vals.count("input_regex"))
        sim_reader->set_files_regex(
            opt_vals["input_regex"].as<string>());

    std::string time_i;
    if (opt_vals.count("start_date"))
        time_i = opt_vals["start_date"].as<string>();

    std::string time_j;
    if (opt_vals.count("end_date"))
        time_j = opt_vals["end_date"].as<string>();

    // some minimal check for mising options
    if (sim_reader->get_file_name().empty()
        && sim_reader->get_files_regex().empty())
    {
        TECA_ERROR(
            "mising file name or regex for simulation reader. "
            "See --help for a list of comand line options.")
        return -1;
    }

    // run the reporting phase of the pipeline
    teca_metadata md = sim_reader->update_metadata();
    //md.to_stream(cerr);

    // extract metadata
    if (!md.has("files"))
    {
        TECA_ERROR("no files were located")
        return -1;
    }
    int n_files = md.get("files")->size();

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
    p_teca_double_array time;
    if (md.get("coordinates", coords)
       || !(time = std::dynamic_pointer_cast<teca_double_array>(coords.get("t"))))
    {
        TECA_ERROR("failed to determine time coordinate")
        return -1;
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
             time, true, calendar, units, time_i, i0))
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
             time, false, calendar, units, time_j, i1))
        {
            TECA_ERROR("Failed to locate time step for end date \""
                << time_j << "\"")
            return -1;
        }
    }

    oss.str("");
    oss << "A total of " <<  n_time_steps << " steps available in " << n_files
        << " files. Using the " << calendar << " calendar. Times are specified in units of "
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
    for (; it != end; ++it, ++i)
    {
        if ((i >= line_len) && (*it == ' '))
        {
            cerr << endl;
            ++it;
            i = 0;
        }
        cerr << *it;
    }
    cerr << endl;

    return 0;
}
