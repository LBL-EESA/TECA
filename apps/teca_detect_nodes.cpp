#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_l2_norm.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_mpi_manager.h"
#include "teca_app_util.h"
#include "teca_detect_nodes.h"

//#include "teca_config.h"
//#include "teca_vorticity.h"
//#include "teca_tc_candidates.h"
//#include "teca_tc_trajectory.h"
//#include "teca_array_collection.h"
//#include "teca_variant_array.h"
//#include "teca_metadata.h"
//#include "teca_table_reader.h"
//#include "teca_table_reduce.h"
//#include "teca_table_sort.h"
//#include "teca_table_calendar.h"
//#include "teca_table_writer.h"
//#include "teca_coordinate_util.h"
//#include "teca_calcalcs.h"
//#include <vector>

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

        ("n_threads", value<int>()->default_value(-1), "\nSets the thread pool"
            " size on each MPI rank. When the default value of -1 is used TECA"
            " will coordinate the thread pools across ranks such each thread"
            " is bound to a unique physical core.\n")

        ("in_connect", value<string>()->default_value(""), "\nConnectivity file\n")
        ("search_by_min", value<string>()->default_value(""),
            "\nVariable to search for the minimum\n")
        ("search_by_max", value<string>()->default_value(""),
            "\nVariable to search for the maximum\n")
        ("closed_contour_cmd", value<string>()->default_value(""),
            "\nClosed contour commands [var,delta,dist,minmaxdist;...]\n")
        ("no_closed_contour_cmd", value<string>()->default_value(""),
            "\nClosed contour commands [var,delta,dist,minmaxdist;...]\n")
        ("threshold_cmd", value<string>()->default_value(""),
            "\nThreshold commands [var,op,value,dist;...]\n")
        ("output_cmd", value<string>()->default_value(""),
            "\nOutput commands [var,op,dist;...]\n")
        ("search_by_threshold", value<string>()->default_value(""),
            "\nThreshold for search operation\n")
        ("min_lon", value<double>()->default_value(0.0),
            "\nMinimum longitude in degrees for detection\n")
        ("max_lon", value<double>()->default_value(10.0),
            "\nMaximum longitude in degrees for detection\n")
        ("min_lat", value<double>()->default_value(-20.0),
            "\nMinimum latitude in degrees for detection\n")
        ("max_lat", value<double>()->default_value(20.0),
            "\nMaximum latitude in degrees for detection\n")
        ("min_abs_lat", value<double>()->default_value(0.0),
            "\nMinimum absolute value of latitude in degrees for detection\n")
        ("merge_dist", value<double>()->default_value(6.0),
            "\nMinimum allowable distance between two candidates in degrees\n")
        ("diag_connect", value<bool>()->default_value(false),
            "\nDiagonal connectivity for RLL grids\n")
        ("regional", value<bool>()->default_value(true),
            "\nRegional (do not wrap longitudinal boundaries)\n")
        ("out_header", value<bool>()->default_value(true),
            "\nOutput header\n")

        ("sea_level_pressure", value<string>()->default_value(""),
            "\nname of variable with sea level pressure\n")
        ("500mb_height", value<string>()->default_value(""),
            "\nname of variable with 500mb height for thickness calc\n")
        ("300mb_height", value<string>()->default_value(""),
            "\nname of variable with 300mb height for thickness calc\n")
        ("surface_wind_u", value<string>()->default_value(""),
            "\nname of variable with surface wind x-component\n")
        ("surface_wind_v", value<string>()->default_value(""),
            "\nname of variable with surface wind y-component\n")

        ("verbose", value<int>()->default_value(0),
            "\nUse 1 to enable verbose mode, otherwise 0.\n")
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

    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    thickness->set_dependent_variables({"Z1000", "Z200"});
    thickness->set_derived_variable("thickness");
    thickness->get_properties_description("thickness", advanced_opt_defs);

    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    surf_wind->set_component_0_variable("UBOT");
    surf_wind->set_component_1_variable("VBOT");
    surf_wind->set_l2_norm_variable("surface_wind");
    surf_wind->get_properties_description("surface_wind_speed", advanced_opt_defs);

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
    thickness->set_properties("thickness", opt_vals);
    surf_wind->set_properties("surface_wind_speed", opt_vals);

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

    if (!opt_vals["search_by_min"].defaulted())
    {
       candidates->set_search_by_min(opt_vals["search_by_min"].as<string>());
    }

    if (!opt_vals["search_by_max"].defaulted())
    {
       candidates->set_search_by_max(opt_vals["search_by_max"].as<string>());
    }

    if (opt_vals["search_by_min"].as<string>() == "" && opt_vals["search_by_max"].as<string>() == "")
    {
       if (opt_vals["sea_level_pressure"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with sea level pressure")
       else
          candidates->set_search_by_min(opt_vals["sea_level_pressure"].as<string>());
    }

    if (!opt_vals["closed_contour_cmd"].defaulted())
    {
       candidates->set_closed_contour_cmd(opt_vals["closed_contour_cmd"].as<string>());
    }
    else
    {
       if (opt_vals["500mb_height"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with 500mb height for thickness calc")
       else
          thickness->set_dependent_variable(0, opt_vals["500mb_height"].as<string>());

       if (opt_vals["300mb_height"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with 300mb height for thickness calc")
       else
          thickness->set_dependent_variable(1, opt_vals["300mb_height"].as<string>());

       size_t n_var = thickness->get_number_of_dependent_variables();
       if (n_var != 2)
       {
          TECA_FATAL_ERROR("thickness calculation requires 2 "
                "variables. given " << n_var)
          return -1;
       }
       thickness->set_execute_callback(point_wise_difference(thickness->get_dependent_variable(0),
                                                             thickness->get_dependent_variable(1),
                                                             thickness->get_derived_variable()));
       std::string text = opt_vals["sea_level_pressure"].as<string>()+",200.0,5.5,0;"+thickness->get_derived_variable()+",-6.0,6.5,1.0";
       candidates->set_closed_contour_cmd(text);
    }

    if (!opt_vals["no_closed_contour_cmd"].defaulted())
    {
       candidates->set_no_closed_contour_cmd(opt_vals["no_closed_contour_cmd"].as<string>());
    }

    if (!opt_vals["threshold_cmd"].defaulted())
    {
       candidates->set_threshold_cmd(opt_vals["threshold_cmd"].as<string>());
    }

    if (!opt_vals["output_cmd"].defaulted())
    {
       candidates->set_output_cmd(opt_vals["output_cmd"].as<string>());
    }
    else
    {
       if (opt_vals["surface_wind_u"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with surface wind x-component")
       else
          surf_wind->set_component_0_variable(opt_vals["surface_wind_u"].as<string>());

       if (opt_vals["surface_wind_v"].as<string>() == "")
          TECA_FATAL_ERROR("Missing name of variable with surface wind y-component")
       else
          surf_wind->set_component_1_variable(opt_vals["surface_wind_v"].as<string>());

       std::string text = opt_vals["sea_level_pressure"].as<string>()+",min,0;"+surf_wind->get_l2_norm_variable()+",max,2";
       candidates->set_output_cmd(text);
    }

    if (!opt_vals["search_by_threshold"].defaulted())
    {
       candidates->set_search_by_threshold(opt_vals["search_by_threshold"].as<string>());
    }

    if (!opt_vals["min_lon"].defaulted())
    {
       candidates->set_min_lon(opt_vals["min_lon"].as<double>());
    }

    if (!opt_vals["max_lon"].defaulted())
    {
       candidates->set_max_lon(opt_vals["max_lon"].as<double>());
    }

    if (!opt_vals["min_lat"].defaulted())
    {
       candidates->set_min_lat(opt_vals["min_lat"].as<double>());
    }

    if (!opt_vals["max_lat"].defaulted())
    {
       candidates->set_max_lat(opt_vals["max_lat"].as<double>());
    }

    if (!opt_vals["min_abs_lat"].defaulted())
    {
       candidates->set_min_abs_lat(opt_vals["min_abs_lat"].as<double>());
    }

    if (!opt_vals["merge_dist"].defaulted())
    {
       candidates->set_merge_dist(opt_vals["merge_dist"].as<double>());
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
    //candidates->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    candidates->set_verbose(opt_vals["verbose"].as<int>());
    candidates->initialize();

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
    thickness->set_input_connection(sim_coords->get_output_port());
    surf_wind->set_input_connection(thickness->get_output_port());
    candidates->set_input_connection(surf_wind->get_output_port());

    // run the pipeline
    candidates->update();

    return 0;
}
