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

        ("candidate_file", value<string>()->default_value("candidates.bin"),
            "\nfile path to write the storm candidates to. The extension determines"
            " the file format. May be one of `.nc`, `.csv`, or `.bin`\n")

        ("850mb_wind_u", value<string>()->default_value("U850"), "\nname of variable with 850 mb wind x-component\n")
        ("850mb_wind_v", value<string>()->default_value("V850"), "\nname of variable with 850 mb wind x-component\n")
        ("surface_wind_u", value<string>()->default_value("UBOT"), "\nname of variable with surface wind x-component\n")
        ("surface_wind_v", value<string>()->default_value("VBOT"), "\nname of variable with surface wind y-component\n")
        ("sea_level_pressure", value<string>()->default_value("PSL"), "\nname of variable with sea level pressure\n")
        ("500mb_temp", value<string>()->default_value("T500"), "\nname of variable with 500mb temperature for warm core calc\n")
        ("200mb_temp", value<string>()->default_value("T200"), "\nname of variable with 200mb temperature for warm core calc\n")
        ("1000mb_height", value<string>()->default_value("Z1000"), "\nname of variable with 1000mb height for thickness calc\n")
        ("200mb_height", value<string>()->default_value("Z200"), "\nname of variable with 200mb height for thickness calc\n")
        ("storm_core_radius", value<double>()->default_value(2.0),
            "\nmaximum number of degrees latitude separationi between vorticity max and pressure min defining a storm\n")
        ("min_vorticity", value<double>()->default_value(1.6e-4, "1.6e-4"), "\nminimum vorticty to be considered a tropical storm\n")
        ("vorticity_window", value<double>()->default_value(7.74446, "7.74446"),
            "\nsize of the search window in degrees. storms core must have a local vorticity max centered on this window\n")
        ("pressure_delta", value<double>()->default_value(400.0), "\nmaximum pressure change within specified radius\n")
        ("pressure_delta_radius", value<double>()->default_value(5.0),
            "\nradius in degrees over which max pressure change is computed\n")
        ("core_temp_delta", value<double>()->default_value(0.8, "0.8"), "\nmaximum core temperature change over the specified radius\n")
        ("core_temp_radius", value<double>()->default_value(5.0), "\nradius in degrees over which max core temperature change is computed\n")
        ("thickness_delta", value<double>()->default_value(50.0), "\nmaximum thickness change over the specified radius\n")
        ("thickness_radius", value<double>()->default_value(4.0), "\nradius in degrees over with max thickness change is computed\n")
        ("lowest_lat", value<double>()->default_value(80), "\nlowest latitude in degrees to search for storms\n")
        ("highest_lat", value<double>()->default_value(80), "\nhighest latitude in degrees to search for storms\n")
        ("max_daily_distance", value<double>()->default_value(1600), "\nmax distance in km that a storm can travel in one day\n")
        ("min_wind_speed", value<double>()->default_value(17.0), "\nminimum peak wind speed to be considered a tropical storm\n")
        ("min_wind_duration", value<double>()->default_value(2.0), "\nnumber of, not necessarily consecutive, days min wind speed sustained\n")

        ("track_file", value<string>()->default_value("tracks.bin"), "\nfile path to"
            " write storm tracks to. The extension determines the file format. May be"
            " one of `.nc`, `.csv`, or `.bin`\n")

        ("first_step", value<long>()->default_value(0), "\nfirst time step to process\n")
        ("last_step", value<long>()->default_value(-1), "\nlast time step to process\n")
        ("start_date", value<std::string>(), "\nThe first time to process in 'Y-M-D h:m:s'"
            " format. Note: There must be a space between the date and time specification\n")
        ("end_date", value<std::string>(), "\nThe last time to process in 'Y-M-D h:m:s' format\n")
        ("n_threads", value<int>()->default_value(-1), "\nSets the thread pool size on each"
            " MPI rank. When the default value of -1 is used TECA will coordinate the thread"
            " pools across ranks such each thread is bound to a unique physical core.\n")

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
        "tropical storms pipeline:\n\n"
        "   (cf / mcf_reader)\n"
        "         \\\n"
        "  (surface_wind_speed)--(850mb_vorticity)--(core_temperature)\n"
        "                                                    /\n"
        " (tracks)--(sort)--(map_reduce)--(candidates)--(thickness)\n"
        "     \\                               /\n"
        " (track_writer)              (candidate_writer)\n\n"
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

    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    surf_wind->set_component_0_variable("UBOT");
    surf_wind->set_component_1_variable("VBOT");
    surf_wind->set_l2_norm_variable("surface_wind");
    surf_wind->get_properties_description("surface_wind_speed", advanced_opt_defs);

    p_teca_vorticity vort_850mb = teca_vorticity::New();
    vort_850mb->set_component_0_variable("U850");
    vort_850mb->set_component_1_variable("V850");
    vort_850mb->set_vorticity_variable("850mb_vorticity");
    vort_850mb->get_properties_description("850mb_vorticity", advanced_opt_defs);

    p_teca_derived_quantity core_temp = teca_derived_quantity::New();
    core_temp->set_dependent_variables({"T500", "T200"});
    core_temp->set_derived_variables({"core_temperature"}, {{}});
    core_temp->get_properties_description("core_temperature", advanced_opt_defs);

    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    thickness->set_dependent_variables({"Z1000", "Z200"});
    thickness->set_derived_variables({"thickness"}, {{}});
    thickness->get_properties_description("thickness", advanced_opt_defs);

    p_teca_tc_candidates candidates = teca_tc_candidates::New();
    candidates->set_surface_wind_speed_variable("surface_wind");
    candidates->set_vorticity_850mb_variable("850mb_vorticity");
    candidates->set_sea_level_pressure_variable("PSL");
    candidates->set_core_temperature_variable("core_temperature");
    candidates->set_thickness_variable("thickness");
    candidates->set_max_core_radius(2.0);
    candidates->set_min_vorticity_850mb(1.6e-4);
    candidates->set_vorticity_850mb_window(7.74446);
    candidates->set_max_pressure_delta(400.0);
    candidates->set_max_pressure_radius(5.0);
    candidates->set_max_core_temperature_delta(0.8);
    candidates->set_max_core_temperature_radius(5.0);
    candidates->set_max_thickness_delta(50.0);
    candidates->set_max_thickness_radius(4.0);
    candidates->set_search_lat_low(-80.0);
    candidates->set_search_lat_high(80.0);
    candidates->get_properties_description("candidates", advanced_opt_defs);

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);

    p_teca_table_writer candidate_writer = teca_table_writer::New();
    candidate_writer->set_output_format_auto();
    candidate_writer->get_properties_description("candidate_writer", advanced_opt_defs);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_index_column("storm_id");
    sort->get_properties_description("sort", advanced_opt_defs);

    p_teca_tc_trajectory tracks = teca_tc_trajectory::New();
    tracks->set_max_daily_distance(1600.0);
    tracks->set_min_wind_speed(17.0);
    tracks->set_min_wind_duration(2.0);
    tracks->get_properties_description("tracks", advanced_opt_defs);

    p_teca_table_calendar calendar = teca_table_calendar::New();
    calendar->get_properties_description("calendar", advanced_opt_defs);

    p_teca_table_writer track_writer = teca_table_writer::New();
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
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    surf_wind->set_properties("surface_wind_speed", opt_vals);
    vort_850mb->set_properties("850mb_vorticity", opt_vals);
    core_temp->set_properties("core_temperature", opt_vals);
    thickness->set_properties("thickness", opt_vals);
    candidates->set_properties("candidates", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    candidate_writer->set_properties("candidate_writer", opt_vals);
    sort->set_properties("sort", opt_vals);
    tracks->set_properties("tracks", opt_vals);
    calendar->set_properties("calendar", opt_vals);
    track_writer->set_properties("track_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence

    // configure the reader
    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");

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

    if (!opt_vals["850mb_wind_u"].defaulted())
        vort_850mb->set_component_0_variable(
            opt_vals["850mb_wind_u"].as<string>());

    if (!opt_vals["850mb_wind_v"].defaulted())
        vort_850mb->set_component_1_variable(
            opt_vals["850mb_wind_v"].as<string>());

    if (!opt_vals["surface_wind_u"].defaulted())
        surf_wind->set_component_0_variable(
            opt_vals["surface_wind_u"].as<string>());

    if (!opt_vals["surface_wind_v"].defaulted())
        surf_wind->set_component_1_variable(
            opt_vals["surface_wind_v"].as<string>());

    if (!opt_vals["500mb_temp"].defaulted())
        core_temp->set_dependent_variable(0,
            opt_vals["500mb_temp"].as<string>());

    if (!opt_vals["200mb_temp"].defaulted())
        core_temp->set_dependent_variable(1,
            opt_vals["200mb_temp"].as<string>());

    if (!opt_vals["1000mb_height"].defaulted())
        thickness->set_dependent_variable(0,
            opt_vals["1000mb_height"].as<string>());

    if (!opt_vals["200mb_height"].defaulted())
        thickness->set_dependent_variable(1,
            opt_vals["200mb_height"].as<string>());

    if (!opt_vals["sea_level_pressure"].defaulted())
        candidates->set_sea_level_pressure_variable(
            opt_vals["sea_level_pressure"].as<string>());

    if (!opt_vals["storm_core_radius"].defaulted())
        candidates->set_max_core_radius(
            opt_vals["storm_core_radius"].as<double>());

    if (!opt_vals["min_vorticity"].defaulted())
        candidates->set_min_vorticity_850mb(
            opt_vals["min_vorticity"].as<double>());

    if (!opt_vals["vorticity_window"].defaulted())
        candidates->set_vorticity_850mb_window(
            opt_vals["vorticity_window"].as<double>());

    if (!opt_vals["pressure_delta"].defaulted())
        candidates->set_max_pressure_delta(
            opt_vals["pressure_delta"].as<double>());

    if (!opt_vals["pressure_delta_radius"].defaulted())
        candidates->set_max_pressure_radius(
            opt_vals["pressure_delta_radius"].as<double>());

    if (!opt_vals["core_temp_delta"].defaulted())
        candidates->set_max_core_temperature_delta(
            opt_vals["core_temp_delta"].as<double>());

    if (!opt_vals["core_temp_radius"].defaulted())
        candidates->set_max_core_temperature_radius(
            opt_vals["core_temp_radius"].as<double>());

    if (!opt_vals["thickness_delta"].defaulted())
        candidates->set_max_thickness_delta(
            opt_vals["thickness_delta"].as<double>());

    if (!opt_vals["thickness_radius"].defaulted())
        candidates->set_max_thickness_radius(
            opt_vals["thickness_radius"].as<double>());

    if (!opt_vals["lowest_lat"].defaulted())
        candidates->set_search_lat_low(
            opt_vals["lowest_lat"].as<double>());

    if (!opt_vals["highest_lat"].defaulted())
        candidates->set_search_lat_high(
            opt_vals["highest_lat"].as<double>());

    if (!opt_vals["first_step"].defaulted())
        map_reduce->set_start_index(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        map_reduce->set_end_index(opt_vals["last_step"].as<long>());

    if (!opt_vals["n_threads"].defaulted())
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        map_reduce->set_thread_pool_size(-1);

    if (!opt_vals["max_daily_distance"].defaulted())
        tracks->set_max_daily_distance(
            opt_vals["max_daily_distance"].as<double>());

    if (!opt_vals["min_wind_speed"].defaulted())
        tracks->set_min_wind_speed(
            opt_vals["min_wind_speed"].as<double>());

    if (!opt_vals["min_wind_duration"].defaulted())
        tracks->set_min_wind_duration(
            opt_vals["min_wind_duration"].as<double>());

    candidate_writer->set_file_name(
        opt_vals["candidate_file"].as<string>());

    track_writer->set_file_name(
        opt_vals["track_file"].as<string>());

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

    // now that command line opts have been parsed we can create
    // the programmable algorithms' functors
    size_t n_var = core_temp->get_number_of_dependent_variables();
    if (n_var != 2)
    {
        TECA_FATAL_ERROR("core temperature calculation requires 2 "
            "variables. given " << n_var)
        return -1;
    }
    core_temp->set_execute_callback(
        point_wise_average(
            core_temp->get_dependent_variable(0),
            core_temp->get_dependent_variable(1),
            core_temp->get_derived_variable(0)));

    n_var = thickness->get_number_of_dependent_variables();
    if (n_var != 2)
    {
        TECA_FATAL_ERROR("thickness calculation requires 2 "
            "variables. given " << n_var)
        return -1;
    }
    thickness->set_execute_callback(
        point_wise_difference(
            thickness->get_dependent_variable(0),
            thickness->get_dependent_variable(1),
            thickness->get_derived_variable(0)));

    // and tell the candidate stage what variables the functors produce
    candidates->set_surface_wind_speed_variable(surf_wind->get_l2_norm_variable());
    candidates->set_vorticity_850mb_variable(vort_850mb->get_vorticity_variable());
    candidates->set_core_temperature_variable(core_temp->get_derived_variable(0));
    candidates->set_thickness_variable(thickness->get_derived_variable(0));

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        teca_metadata md = reader->update_metadata();

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
        p_teca_variant_array time;
        if (md.get("coordinates", coords) || !(time = coords.get("t")))
        {
            TECA_FATAL_ERROR("failed to determine time coordinate")
            return -1;
        }

        // convert date string to step, start date
        if (parse_start_date)
        {
            unsigned long first_step = 0;
            std::string start_date = opt_vals["start_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, true, true, calendar,
                 units, start_date, first_step))
            {
                TECA_FATAL_ERROR("Failed to lcoate time step for start date \""
                    <<  start_date << "\"")
                return -1;
            }
            map_reduce->set_start_index(first_step);
        }

        // and end date
        if (parse_end_date)
        {
            unsigned long last_step = 0;
            std::string end_date = opt_vals["end_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, false, true, calendar,
                 units, end_date, last_step))
            {
                TECA_FATAL_ERROR("Failed to lcoate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            map_reduce->set_end_index(last_step);
        }
    }

    // connect all the stages
    sim_coords->set_input_connection(reader->get_output_port());
    surf_wind->set_input_connection(sim_coords->get_output_port());
    vort_850mb->set_input_connection(surf_wind->get_output_port());
    core_temp->set_input_connection(vort_850mb->get_output_port());
    thickness->set_input_connection(core_temp->get_output_port());
    candidates->set_input_connection(thickness->get_output_port());
    map_reduce->set_input_connection(candidates->get_output_port());
    candidate_writer->set_input_connection(map_reduce->get_output_port());
    sort->set_input_connection(candidate_writer->get_output_port());
    tracks->set_input_connection(sort->get_output_port());
    calendar->set_input_connection(tracks->get_output_port());
    track_writer->set_input_connection(calendar->get_output_port());

    // run the pipeline
    track_writer->update();

    return 0;
}
