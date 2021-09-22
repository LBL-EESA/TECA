#include "teca_config.h"
#include "teca_cf_reader.h"
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
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_writer.h"
#include "teca_mpi_manager.h"
#include "teca_coordinate_util.h"
#include "teca_calcalcs.h"
#include "teca_app_util.h"

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
        ("candidate_file", value<string>()->default_value("candidates.bin"), "\nfile path to read the storm candidates from\n")
        ("max_daily_distance", value<double>()->default_value(1600), "\nmax distance in km that a storm can travel in one day\n")
        ("min_wind_speed", value<double>()->default_value(17.0), "\nminimum peak wind speed to be considered a tropical storm\n")
        ("min_wind_duration", value<double>()->default_value(2.0), "\nnumber of, not necessarily consecutive, days min wind speed sustained\n")
        ("track_file", value<string>()->default_value("tracks.bin"), "\nfile path to"
            " write storm tracks to. The extension determines the file format. May be"
            " one of `.nc`, `.csv`, or `.bin`\n")
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
        "tropical storms trajectory pipeline:\n\n"
        "   (candidate reader)--(sort)--(tracks)--(track writer)\n\n"
        "Advanced command line options", help_width, help_width - 4
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_table_reader candidate_reader = teca_table_reader::New();
    candidate_reader->get_properties_description("candidate_reader", advanced_opt_defs);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(candidate_reader->get_output_port());
    sort->set_index_column("storm_id");
    sort->get_properties_description("sort", advanced_opt_defs);

    p_teca_tc_trajectory tracks = teca_tc_trajectory::New();
    tracks->set_input_connection(sort->get_output_port());
    tracks->set_max_daily_distance(1600.0);
    tracks->set_min_wind_speed(17.0);
    tracks->set_min_wind_duration(2.0);
    tracks->get_properties_description("tracks", advanced_opt_defs);

    p_teca_table_calendar calendar = teca_table_calendar::New();
    calendar->set_input_connection(tracks->get_output_port());
    calendar->get_properties_description("calendar", advanced_opt_defs);

    p_teca_table_writer track_writer = teca_table_writer::New();
    track_writer->set_input_connection(calendar->get_output_port());
    track_writer->set_output_format_auto();
    track_writer->get_properties_description("track_writer", advanced_opt_defs);

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
    candidate_reader->set_properties("candidate_reader", opt_vals);
    sort->set_properties("sort", opt_vals);
    tracks->set_properties("tracks", opt_vals);
    calendar->set_properties("calendar", opt_vals);
    track_writer->set_properties("track_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    candidate_reader->set_file_name(
        opt_vals["candidate_file"].as<string>());

    if (!opt_vals["max_daily_distance"].defaulted())
        tracks->set_max_daily_distance(
            opt_vals["max_daily_distance"].as<double>());

    if (!opt_vals["min_wind_speed"].defaulted())
        tracks->set_min_wind_speed(
            opt_vals["min_wind_speed"].as<double>());

    if (!opt_vals["min_wind_duration"].defaulted())
        tracks->set_min_wind_duration(
            opt_vals["min_wind_duration"].as<double>());

    track_writer->set_file_name(
        opt_vals["track_file"].as<string>());

    // some minimal check for missing options
    if (candidate_reader->get_file_name().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("missing file name for candidate reader. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    // run the pipeline
    track_writer->update();

    return 0;
}
