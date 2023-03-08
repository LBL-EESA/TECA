#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_l2_norm.h"
#include "teca_vorticity.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_tc_candidates.h"
#include "teca_tc_trajectory.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_table_reader.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_writer.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_app_util.h"
#include "teca_calcalcs.h"
#include "teca_detect_nodes.h"

#include <vector>
#include <string>
#include <iostream>
#include <boost/program_options.hpp>


using namespace std;
using namespace teca_derived_quantity_numerics;

using boost::program_options::value;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // initialize mpi
    teca_mpi_manager mpi_man(argc, argv);

    // initialize command line options description
    // set up some common options to simplify use for most
    // common scenarios
    int help_width = 100;
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most commonly used. Information\n"
        "on advanced options can be displayed using --advanced_help\n\n"
        "Basic command line options", help_width, help_width - 4
        );
    basic_opt_defs.add_options()
        ("input_file", value<std::string>(), "\na teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\na teca_cf_reader regex identifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nName of the variable to use for x-coordinates.\n")

        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nName of the variable to use for y-coordinates.\n")

        ("in_connect", value<string>()->default_value(""), "\nConnectivity file\n")
        ("searchbymin", value<string>()->default_value("MSL"), "\nVariable to search for the minimum\n")
        ("searchbymax", value<string>()->default_value(""), "\nVariable to search for the maximum\n")
        ("closedcontourcmd", value<string>()->default_value(""), "\nClosed contour commands [var,delta,dist,minmaxdist;...]\n")
        ("noclosedcontourcmd", value<string>()->default_value("MSL,200.0,5.5,0"), "\nClosed contour commands [var,delta,dist,minmaxdist;...]\n")
        ("thresholdcmd", value<string>()->default_value(""), "\nThreshold commands [var,op,value,dist;...]\n")
        ("outputcmd", value<string>()->default_value("MSL,min,0"), "\nOutput commands [var,op,dist;...]\n")
        ("searchbythreshold", value<string>()->default_value(""), "\nThreshold for search operation\n")
        ("minlon", value<double>()->default_value(0.0), "\nMinimum longitude in degrees for detection\n")
        ("maxlon", value<double>()->default_value(10.0), "\nMaximum longitude in degrees for detection\n")
        ("minlat", value<double>()->default_value(0.0), "\nMinimum latitude in degrees for detection\n")
        ("maxlat", value<double>()->default_value(9.0), "\nMaximum latitude in degrees for detection\n")
        ("minabslat", value<double>()->default_value(0.0), "\nMinimum absolute value of latitude in degrees for detection\n")
        ("mergedist", value<double>()->default_value(6.0), "\nMerge distance in degrees\n")
        ("diag_connect", value<bool>()->default_value(false), "\nDiagonal connectivity for RLL grids\n")
        ("regional", value<bool>()->default_value(true), "\nRegional (do not wrap longitudinal boundaries)\n")
        ("out_header", value<bool>()->default_value(true), "\nOutput header\n")

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
        "Advanced command line options", help_width, help_width - 4
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates sim_coords = teca_normalize_coordinates::New();

    p_teca_detect_nodes candidates = teca_detect_nodes::New();
    candidates->get_properties_description("candidates", advanced_opt_defs);

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

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    candidates->set_properties("candidates", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence

    // configure the reader
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");

    if (!opt_vals["x_axis_variable"].defaulted())
    {
        cf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<string>());
        mcf_reader->set_x_axis_variable(opt_vals["x_axis_variable"].as<string>());
    }

    if (!opt_vals["y_axis_variable"].defaulted())
    {
        cf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<string>());
        mcf_reader->set_y_axis_variable(opt_vals["y_axis_variable"].as<string>());
    }

    p_teca_algorithm reader;
    if (opt_vals.count("input_file"))
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        reader = mcf_reader;
    }
    else if (opt_vals.count("input_regex"))
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        reader = cf_reader;
    }

    if (!opt_vals["in_connect"].defaulted())
    {
        candidates->set_in_connect(opt_vals["in_connect"].as<string>());
    }

    if (!opt_vals["searchbymin"].defaulted())
    {
       candidates->set_searchbymin(opt_vals["searchbymin"].as<string>());
    }

    if (!opt_vals["searchbymax"].defaulted())
    {
       candidates->set_searchbymax(opt_vals["searchbymax"].as<string>());
    }

    if (!opt_vals["closedcontourcmd"].defaulted())
    {
       candidates->set_closedcontourcmd(opt_vals["closedcontourcmd"].as<string>());
    }

    if (!opt_vals["noclosedcontourcmd"].defaulted())
    {
       candidates->set_noclosedcontourcmd(opt_vals["noclosedcontourcmd"].as<string>());
    }

    if (!opt_vals["thresholdcmd"].defaulted())
    {
       candidates->set_thresholdcmd(opt_vals["thresholdcmd"].as<string>());
    }

    if (!opt_vals["outputcmd"].defaulted())
    {
       candidates->set_outputcmd(opt_vals["outputcmd"].as<string>());
    }

    if (!opt_vals["searchbythreshold"].defaulted())
    {
       candidates->set_searchbythreshold(opt_vals["searchbythreshold"].as<string>());
    }

    if (!opt_vals["minlon"].defaulted())
    {
       candidates->set_minlon(opt_vals["minlon"].as<double>());
    }

    if (!opt_vals["maxlon"].defaulted())
    {
       candidates->set_maxlon(opt_vals["maxlon"].as<double>());
    }

    if (!opt_vals["minlat"].defaulted())
    {
       candidates->set_minlat(opt_vals["minlat"].as<double>());
    }

    if (!opt_vals["maxlat"].defaulted())
    {
       candidates->set_maxlat(opt_vals["maxlat"].as<double>());
    }

    if (!opt_vals["minabslat"].defaulted())
    {
       candidates->set_minabslat(opt_vals["minabslat"].as<double>());
    }

    if (!opt_vals["mergedist"].defaulted())
    {
       candidates->set_mergedist(opt_vals["mergedist"].as<double>());
    }

    if (!opt_vals["diag_connect"].defaulted())
    {
       candidates->set_diag_connect(opt_vals["diag_connect"].as<bool>());
    }

    if (!opt_vals["regional"].defaulted())
    {
       candidates->set_regional(opt_vals["regional"].as<bool>());
    }

    if (!opt_vals["out_header"].defaulted())
    {
       candidates->set_out_header(opt_vals["out_header"].as<bool>());
    }

    // some minimal check for missing options
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    // connect all the stages
    sim_coords->set_input_connection(reader->get_output_port());
    candidates->set_input_connection(sim_coords->get_output_port());

    // run the pipeline
    candidates->update();

    return 0;
}
