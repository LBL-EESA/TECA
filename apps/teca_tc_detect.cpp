#include "teca_config.h"
#include "teca_cf_reader.h"
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
#include "calcalcs.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>

using namespace std;
using namespace teca_derived_quantity_numerics;

using boost::program_options::value;

using seconds_t =
    std::chrono::duration<double, std::chrono::seconds::period>;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // initialize mpi
    teca_mpi_manager mpi_man(argc, argv);

    std::chrono::high_resolution_clock::time_point t0, t1;
    if (mpi_man.get_comm_rank() == 0)
        t0 = std::chrono::high_resolution_clock::now();

    // initialize command line options description
    // set up some common options to simplify use for most
    // common scenarios
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most commonly used. Information\n"
        "on advanced options can be displayed using --advanced_help\n\n"
        "Basic command line options", 120, -1
        );
    basic_opt_defs.add_options()
        ("input_file", value<string>(), "file path to the simulation to search for tropical cyclones")
        ("input_regex", value<string>(), "regex matching simulation files to search for tropical cylones")
        ("candidate_file", value<string>(), "file path to write the storm candidates to (candidates.bin)")
        ("850mb_wind_u", value<string>(), "name of variable with 850 mb wind x-component (U850)")
        ("850mb_wind_v", value<string>(), "name of variable with 850 mb wind x-component (V850)")
        ("surface_wind_u", value<string>(), "name of variable with surface wind x-component (UBOT)")
        ("surface_wind_v", value<string>(), "name of variable with surface wind y-component (VBOT)")
        ("sea_level_pressure", value<string>(), "name of variable with sea level pressure (PSL)")
        ("500mb_temp", value<string>(), "name of variable with 500mb temperature for warm core calc (T500)")
        ("200mb_temp", value<string>(), "name of variable with 200mb temperature for warm core calc (T200)")
        ("1000mb_height", value<string>(), "name of variable with 1000mb height for thickness calc (Z1000)")
        ("200mb_height", value<string>(), "name of variable with 200mb height for thickness calc (Z200)")
        ("storm_core_radius", value<double>(), "maximum number of degrees latitude separation between vorticity max and pressure min defining a storm (2.0)")
        ("min_vorticity", value<double>(), "minimum vorticty to be considered a tropical storm (1.6e-4)")
        ("vorticity_window", value<double>(), "size of the search window in degrees. storms core must have a local vorticity max centered on this window (7.74446)")
        ("pressure_delta", value<double>(), "maximum pressure change within specified radius (400.0)")
        ("pressure_delta_radius", value<double>(), "radius in degrees over which max pressure change is computed (5.0)")
        ("core_temp_delta", value<double>(), "maximum core temperature change over the specified radius (0.8)")
        ("core_temp_radius", value<double>(), "radius in degrees over which max core temperature change is computed (5.0)")
        ("thickness_delta", value<double>(), "maximum thickness change over the specified radius (50.0)")
        ("thickness_radius", value<double>(), "radius in degrees over with max thickness change is comuted (4.0)")
        ("lowest_lat", value<double>(), "lowest latitude in degrees to search for storms (80)")
        ("highest_lat", value<double>(), "highest latitude in degrees to search for storms (80)")
        ("max_daily_distance", value<double>(), "max distance in km that a storm can travel in one day (1600)")
        ("min_wind_speed", value<double>(), "minimum peak wind speed to be considered a tropical storm (17.0)")
        ("min_wind_duration", value<double>(), "number of, not necessarily consecutive, days min wind speed sustained (2.0)")
        ("track_file", value<string>(), "file path to write storm tracks to")
        ("first_step", value<long>(), "first time step to process")
        ("last_step", value<long>(), "last time step to process")
        ("start_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("end_date", value<string>(), "first time to proces in YYYY-MM-DD hh:mm:ss format")
        ("n_threads", value<int>(), "thread pool size. default is 1. -1 for all")
        ("help", "display the basic options help")
        ("advanced_help", "display the advanced options help")
        ("full_help", "display entire help message")
        ;

    // add all options from each pipeline stage for more advanced use
    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "tropical storms pipeline:\n\n"
        "   (sim_reader)\n"
        "        \\\n"
        "  (surface_wind_speed)--(850mb_vorticity)--(core_temperature)\n"
        "                                                    /\n"
        " (tracks)--(sort)--(map_reduce)--(candidates)--(thickness)\n"
        "     \\                               /\n"
        " (track_writer)              (candidate_writer)\n\n"
        "Advanced command line options", -1, 1
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_cf_reader sim_reader = teca_cf_reader::New();
    sim_reader->get_properties_description("sim_reader", advanced_opt_defs);

    p_teca_l2_norm surf_wind = teca_l2_norm::New();
    surf_wind->set_input_connection(sim_reader->get_output_port());
    surf_wind->set_component_0_variable("UBOT");
    surf_wind->set_component_1_variable("VBOT");
    surf_wind->set_l2_norm_variable("surface_wind");
    surf_wind->get_properties_description("surface_wind_speed", advanced_opt_defs);

    p_teca_vorticity vort_850mb = teca_vorticity::New();
    vort_850mb->set_input_connection(surf_wind->get_output_port());
    vort_850mb->set_component_0_variable("U850");
    vort_850mb->set_component_1_variable("V850");
    vort_850mb->set_vorticity_variable("850mb_vorticity");
    vort_850mb->get_properties_description("850mb_vorticity", advanced_opt_defs);

    p_teca_derived_quantity core_temp = teca_derived_quantity::New();
    core_temp->set_input_connection(vort_850mb->get_output_port());
    core_temp->set_dependent_variables({"T500", "T200"});
    core_temp->set_derived_variable("core_temperature");
    core_temp->get_properties_description("core_temperature", advanced_opt_defs);

    p_teca_derived_quantity thickness = teca_derived_quantity::New();
    thickness->set_input_connection(core_temp->get_output_port());
    thickness->set_dependent_variables({"Z1000", "Z200"});
    thickness->set_derived_variable("thickness");
    thickness->get_properties_description("thickness", advanced_opt_defs);

    p_teca_tc_candidates candidates = teca_tc_candidates::New();
    candidates->set_input_connection(thickness->get_output_port());
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
    map_reduce->set_input_connection(candidates->get_output_port());
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);

    p_teca_table_writer candidate_writer = teca_table_writer::New();
    candidate_writer->set_input_connection(map_reduce->get_output_port());
    candidate_writer->set_output_format_auto();
    candidate_writer->get_properties_description("candidate_writer", advanced_opt_defs);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(candidate_writer->get_output_port());
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
    options_description all_opt_defs(-1, -1);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(all_opt_defs).run(),
            opt_vals);

        if (mpi_man.get_comm_rank() == 0)
        {
            if (opt_vals.count("help"))
            {
                cerr << endl
                    << "usage: teca_tc_detect [options]" << endl
                    << endl
                    << basic_opt_defs << endl
                    << endl;
                return -1;
            }
            if (opt_vals.count("advanced_help"))
            {
                cerr << endl
                    << "usage: teca_tc_detect [options]" << endl
                    << endl
                    << advanced_opt_defs << endl
                    << endl;
                return -1;
            }

            if (opt_vals.count("full_help"))
            {
                cerr << endl
                    << "usage: teca_tc_detect [options]" << endl
                    << endl
                    << all_opt_defs << endl
                    << endl;
                return -1;
            }
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        TECA_ERROR("Error parsing command line options. See --help "
            "for a list of supported options. " << e.what())
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    sim_reader->set_properties("sim_reader", opt_vals);
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
    if (opt_vals.count("input_file"))
        sim_reader->set_file_name(
            opt_vals["input_file"].as<string>());

    if (opt_vals.count("input_regex"))
        sim_reader->set_files_regex(
            opt_vals["input_regex"].as<string>());

    if (opt_vals.count("850mb_wind_u"))
        vort_850mb->set_component_0_variable(
            opt_vals["850mb_wind_u"].as<string>());

    if (opt_vals.count("850mb_wind_v"))
        vort_850mb->set_component_1_variable(
            opt_vals["850mb_wind_v"].as<string>());

    if (opt_vals.count("surface_wind_u"))
        surf_wind->set_component_0_variable(
            opt_vals["surface_wind_u"].as<string>());

    if (opt_vals.count("surface_wind_v"))
        surf_wind->set_component_1_variable(
            opt_vals["surface_wind_v"].as<string>());

    std::vector<std::string> dep_var;
    core_temp->get_dependent_variables(dep_var);
    if (opt_vals.count("500mb_temp"))
        dep_var[0] = opt_vals["500mb_temp"].as<string>();
    if (opt_vals.count("200mb_temp"))
        dep_var[1] = opt_vals["200mb_temp"].as<string>();
    core_temp->set_dependent_variables(dep_var);
    dep_var.clear();

    thickness->get_dependent_variables(dep_var);
    if (opt_vals.count("1000mb_height"))
        dep_var[0] = opt_vals["1000mb_height"].as<string>();
    if (opt_vals.count("200mb_height"))
        dep_var[1] = opt_vals["200mb_height"].as<string>();
    thickness->set_dependent_variables(dep_var);
    dep_var.clear();

    if (opt_vals.count("sea_level_pressure"))
        candidates->set_sea_level_pressure_variable(
            opt_vals["sea_level_pressure"].as<string>());

    if (opt_vals.count("storm_core_radius"))
        candidates->set_max_core_radius(
            opt_vals["storm_core_radius"].as<double>());

    if (opt_vals.count("min_vorticity"))
        candidates->set_min_vorticity_850mb(
            opt_vals["min_vorticity"].as<double>());

    if (opt_vals.count("vorticity_window"))
        candidates->set_vorticity_850mb_window(
            opt_vals["vorticity_window"].as<double>());

    if (opt_vals.count("pressure_delta"))
        candidates->set_max_pressure_delta(
            opt_vals["pressure_delta"].as<double>());

    if (opt_vals.count("pressure_delta_radius"))
        candidates->set_max_pressure_radius(
            opt_vals["pressure_delta_radius"].as<double>());

    if (opt_vals.count("core_temp_delta"))
        candidates->set_max_core_temperature_delta(
            opt_vals["core_temp_delta"].as<double>());

    if (opt_vals.count("core_temp_radius"))
        candidates->set_max_core_temperature_radius(
            opt_vals["core_temp_radius"].as<double>());

    if (opt_vals.count("thickness_delta"))
        candidates->set_max_thickness_delta(
            opt_vals["thickness_delta"].as<double>());

    if (opt_vals.count("thickness_radius"))
        candidates->set_max_thickness_radius(
            opt_vals["thickness_radius"].as<double>());

    if (opt_vals.count("lowest_lat"))
        candidates->set_search_lat_low(
            opt_vals["lowest_lat"].as<double>());

    if (opt_vals.count("highest_lat"))
        candidates->set_search_lat_high(
            opt_vals["highest_lat"].as<double>());

    if (opt_vals.count("first_step"))
        map_reduce->set_first_step(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        map_reduce->set_last_step(opt_vals["last_step"].as<long>());

    if (opt_vals.count("n_threads"))
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    if (opt_vals.count("max_daily_distance"))
        tracks->set_max_daily_distance(
            opt_vals["max_daily_distance"].as<double>());

    if (opt_vals.count("min_wind_speed"))
        tracks->set_min_wind_speed(
            opt_vals["min_wind_speed"].as<double>());

    if (opt_vals.count("min_wind_duration"))
        tracks->set_min_wind_duration(
            opt_vals["min_wind_duration"].as<double>());

    if (opt_vals.count("candidate_file"))
        candidate_writer->set_file_name(
            opt_vals["candidate_file"].as<string>());

    if (opt_vals.count("track_file"))
        track_writer->set_file_name(
            opt_vals["track_file"].as<string>());


    // some minimal check for missing options
    if (sim_reader->get_file_name().empty()
        && sim_reader->get_files_regex().empty())
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR(
                "missing file name or regex for simulation reader. "
                "See --help for a list of command line options.")
        }
        return -1;
    }

    // now that command line opts have been parsed we can create
    // the programmable algorithms' functors
    core_temp->get_dependent_variables(dep_var);
    if (dep_var.size() != 2)
    {
        TECA_ERROR("core temperature calculation requires 2 "
            "variables. given " << dep_var.size())
        return -1;
    }
    core_temp->set_execute_callback(
        point_wise_average(dep_var[0], dep_var[1],
        core_temp->get_derived_variable()));
    dep_var.clear();

    thickness->get_dependent_variables(dep_var);
    if (dep_var.size() != 2)
    {
        TECA_ERROR("thickness calculation requires 2 "
            "variables. given " << dep_var.size())
        return -1;
    }
    thickness->set_execute_callback(
        point_wise_difference(dep_var[0], dep_var[1],
        thickness->get_derived_variable()));
    dep_var.clear();

    // and tell the candidate stage what variables the functors produce
    candidates->set_surface_wind_speed_variable(surf_wind->get_l2_norm_variable());
    candidates->set_vorticity_850mb_variable(vort_850mb->get_vorticity_variable());
    candidates->set_core_temperature_variable(core_temp->get_derived_variable());
    candidates->set_thickness_variable(thickness->get_derived_variable());

    // look for requested time step range, start
    bool parse_start_date = opt_vals.count("start_date");
    bool parse_end_date = opt_vals.count("end_date");
    if (parse_start_date || parse_end_date)
    {
        // run the reporting phase of the pipeline
        teca_metadata md = sim_reader->update_metadata();

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
        if (md.get("coordinates", coords) ||
            !(time = std::dynamic_pointer_cast<teca_double_array>(
                coords.get("t"))))
        {
            TECA_ERROR("failed to determine time coordinate")
            return -1;
        }

        // convert date string to step, start date
        if (parse_start_date)
        {
            unsigned long first_step = 0;
            std::string start_date = opt_vals["start_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, true, calendar,
                 units, start_date, first_step))
            {
                TECA_ERROR("Failed to lcoate time step for start date \""
                    <<  start_date << "\"")
                return -1;
            }
            map_reduce->set_first_step(first_step);
        }

        // and end date
        if (parse_end_date)
        {
            unsigned long last_step = 0;
            std::string end_date = opt_vals["end_date"].as<string>();
            if (teca_coordinate_util::time_step_of(time, false, calendar,
                 units, end_date, last_step))
            {
                TECA_ERROR("Failed to lcoate time step for end date \""
                    <<  end_date << "\"")
                return -1;
            }
            map_reduce->set_last_step(last_step);
        }
    }

    // run the pipeline
    track_writer->update();

    if (mpi_man.get_comm_rank() == 0)
    {
        t1 = std::chrono::high_resolution_clock::now();
        seconds_t dt(t1 - t0);
        TECA_STATUS("teca_tc_detect run_time=" << dt.count() << " sec")
    }

    return 0;
}
