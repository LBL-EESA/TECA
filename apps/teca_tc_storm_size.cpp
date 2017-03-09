#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_remove_rows.h"
#include "teca_table_sort.h"
#include "teca_cf_reader.h"
#include "teca_tc_storm_size.h"
#include "teca_table_reduce.h"
#include "teca_table_to_stream.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_time_step_executive.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include <boost/program_options.hpp>

using std::cerr;
using std::endl;

using seconds_t =
    std::chrono::duration<double, std::chrono::seconds::period>;

using boost::program_options::value;


int main(int argc, char **argv)
{
    // initialize MPI
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
        ("track_file", value<std::string>(), "file path to read the cyclone from (tracks.bin)")
        ("wind_files", value<std::string>(), "regex matching simulation files containing wind fields ()")
        ("track_file_out", value<std::string>(), "file path to write cyclone tracks with size (tracks_size.bin)")
        ("wind_u_var", value<std::string>(), "name of variable with wind x-component (UBOT)")
        ("wind_v_var", value<std::string>(), "name of variable with wind y-component (VBOT)")
        ("track_mask", value<std::string>(), "expression to filter tracks by ()")
        ("number_of_bins", value<int>(), "number of bins in the radial wind decomposition (32)")
        ("profile_type", value<std::string>(), "radial wind profile type. max or avg (avg)")
        ("search_radius", value<double>(), "size of search window in deg lat (6)")
        ("first_track", value<long>(), "first track to process")
        ("last_track", value<long>(), "last track to process")
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
        "tc storm size pipeline:\n\n"
        "   (track reader)--(track filter)\n"
        "                        \\\n"
        "     (wind reader)--(storm size)\n"
        "                         \\\n"
        "                      (map reduce)--(table sort)\n"
        "                                          \\\n"
        "                                       (track writer)\n\n"
        "Advanced command line options", -1, 1
        );

    // create the pipeline stages here, they contain the
    // documentation and parse command line.
    // objects report all of their properties directly
    // set default options here so that command line options override
    // them. while we are at it connect the pipeline
    p_teca_table_reader track_reader = teca_table_reader::New();
    track_reader->get_properties_description("track_reader", advanced_opt_defs);
    track_reader->set_file_name("tracks.bin");

    p_teca_table_remove_rows track_filter = teca_table_remove_rows::New();
    track_filter->get_properties_description("track_filter", advanced_opt_defs);

    p_teca_cf_reader wind_reader = teca_cf_reader::New();
    wind_reader->get_properties_description("wind_reader", advanced_opt_defs);

    p_teca_tc_storm_size storm_size = teca_tc_storm_size::New();
    storm_size->get_properties_description("storm_size", advanced_opt_defs);
    storm_size->set_input_connection(1, wind_reader->get_output_port());


    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);
    map_reduce->set_input_connection(storm_size->get_output_port());

    p_teca_table_sort sort = teca_table_sort::New();
    sort->get_properties_description("table_sort", advanced_opt_defs);
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("track_id");
    sort->enable_stable_sort();

    p_teca_table_writer track_writer = teca_table_writer::New();
    track_writer->get_properties_description("track_writer", advanced_opt_defs);
    track_writer->set_input_connection(sort->get_output_port());
    track_writer->set_file_name("tracks_size.bin");

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
                    << "usage: teca_tc_storm_size [options]" << endl
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
    track_reader->set_properties("track_reader", opt_vals);
    track_filter->set_properties("track_filter", opt_vals);
    wind_reader->set_properties("wind_reader", opt_vals);
    storm_size->set_properties("storm_size", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    sort->set_properties("table_sort", opt_vals);
    track_writer->set_properties("track_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    if (opt_vals.count("track_file"))
        track_reader->set_file_name(opt_vals["track_file"].as<std::string>());

    if (opt_vals.count("wind_files"))
    {
        wind_reader->set_files_regex(opt_vals["wind_files"].as<std::string>());
    }
    else
    {
        TECA_ERROR("--wind_files is a required option")
        return -1;
    }

    if (opt_vals.count("track_file_out"))
        track_writer->set_file_name(opt_vals["track_file_out"].as<std::string>());

    if (opt_vals.count("track_mask"))
    {
        track_filter->set_input_connection(track_reader->get_output_port());
        track_filter->set_mask_expression(opt_vals["track_mask"].as<std::string>());
        storm_size->set_input_connection(0, track_filter->get_output_port());
    }
    else
    {
        storm_size->set_input_connection(0, track_reader->get_output_port());
    }

    if (opt_vals.count("wind_u_var"))
        storm_size->set_wind_u_variable(opt_vals["wind_u_var"].as<std::string>());

    if (opt_vals.count("wind_v_var"))
        storm_size->set_wind_v_variable(opt_vals["wind_v_var"].as<std::string>());

    if (opt_vals.count("n_radial_bins"))
        storm_size->set_number_of_radial_bins(opt_vals["n_radial_bins"].as<int>());

    if (opt_vals.count("profile_type"))
    {
        std::string profile_type = opt_vals["profile_type"].as<std::string>();
        if (profile_type == "avg")
        {
            storm_size->set_profile_type(1);
        }
        else if (profile_type == "max")
        {
            storm_size->set_profile_type(0);
        }
        else
        {
            TECA_ERROR("invalid profile_type " << profile_type)
            return -1;
        }
    }

    if (opt_vals.count("search_radius"))
        storm_size->set_search_radius(opt_vals["search_radius"].as<double>());

    if (opt_vals.count("first_track"))
        map_reduce->set_first_step(opt_vals["first_track"].as<long>());

    if (opt_vals.count("last_track"))
        map_reduce->set_last_step(opt_vals["last_track"].as<long>());

    if (opt_vals.count("n_threads"))
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    // run the pipeline
    track_writer->update();

    if (mpi_man.get_comm_rank() == 0)
    {
        t1 = std::chrono::high_resolution_clock::now();
        seconds_t dt(t1 - t0);
        TECA_STATUS("teca_tc_storm_size run_time=" << dt.count() << " sec")
    }

    return 0;
}
