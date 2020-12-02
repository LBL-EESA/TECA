#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_remove_rows.h"
#include "teca_table_sort.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_tc_wind_radii.h"
#include "teca_table_reduce.h"
#include "teca_table_to_stream.h"
#include "teca_table_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_app_util.h"
#include "teca_mpi_manager.h"

#include <vector>
#include <string>
#include <iostream>

#include <boost/program_options.hpp>

using std::cerr;
using std::endl;

using boost::program_options::value;


int main(int argc, char **argv)
{
    // initialize MPI
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
        ("track_file", value<std::string>(), "\nA file containing cyclone tracks (tracks.bin)\n")

        ("input_file", value<std::string>(), "\na teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\na teca_cf_reader regex identifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("wind_files", value<std::string>(), "\na synonym for --input_regex.\n")

        ("track_file_out", value<std::string>(), "\nfile path to write cyclone tracks with size (tracks_size.bin)\n")
        ("wind_u_var", value<std::string>(), "\nname of variable with wind x-component (UBOT)\n")
        ("wind_v_var", value<std::string>(), "\nname of variable with wind y-component (VBOT)\n")
        ("track_mask", value<std::string>(), "\nAn expression to filter tracks by ()\n")
        ("number_of_bins", value<int>(), "\nnumber of bins in the radial wind decomposition (32)\n")
        ("profile_type", value<std::string>(), "\nradial wind profile type. max or avg (avg)\n")
        ("search_radius", value<double>(), "\nsize of search window in degrees (6)\n")
        ("first_track", value<long>(), "\nfirst track to process\n")
        ("last_track", value<long>(), "\nlast track to process\n")
        ("n_threads", value<int>(), "\nSets the thread pool size on each MPI rank. When the default"
            " value of -1 is used TECA will coordinate the thread pools across ranks such each"
            " thread is bound to a unique physical core.\n")
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
        "tc storm size pipeline:\n\n"
        "   (track reader)--(track filter)\n"
        "                        \\\n"
        "  (cf / mcf_reader)--(storm size)\n"
        "                         \\\n"
        "                      (map reduce)--(table sort)\n"
        "                                          \\\n"
        "                                       (track writer)\n\n"
        "Advanced command line options", help_width, help_width - 4
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

    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    mcf_reader->get_properties_description("mcf_reader", advanced_opt_defs);

    p_teca_normalize_coordinates wind_coords = teca_normalize_coordinates::New();

    p_teca_tc_wind_radii wind_radii = teca_tc_wind_radii::New();
    wind_radii->get_properties_description("wind_radii", advanced_opt_defs);

    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);

    p_teca_table_sort sort = teca_table_sort::New();
    sort->get_properties_description("table_sort", advanced_opt_defs);
    sort->set_index_column("track_id");
    sort->enable_stable_sort();

    p_teca_table_writer track_writer = teca_table_writer::New();
    track_writer->get_properties_description("track_writer", advanced_opt_defs);
    track_writer->set_file_name("tracks_size.bin");

    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    if (teca_app_util::process_command_line_help(mpi_man.get_comm_rank(),
        argc, argv, basic_opt_defs, advanced_opt_defs, all_opt_defs, opt_vals))
    {
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    track_reader->set_properties("track_reader", opt_vals);
    track_filter->set_properties("track_filter", opt_vals);
    cf_reader->set_properties("cf_reader", opt_vals);
    mcf_reader->set_properties("mcf_reader", opt_vals);
    wind_radii->set_properties("wind_radii", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    sort->set_properties("table_sort", opt_vals);
    track_writer->set_properties("track_writer", opt_vals);

    // now pass in the basic options, these are processed
    // last so that they will take precedence
    if (opt_vals.count("track_file"))
        track_reader->set_file_name(opt_vals["track_file"].as<std::string>());

    bool have_file = opt_vals.count("input_file");
    bool have_wind_files = opt_vals.count("wind_files");
    bool have_regex = opt_vals.count("input_regex");
    p_teca_algorithm wind_reader;
    if (have_file)
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<std::string>());
        wind_reader = mcf_reader;
    }
    else if (have_wind_files)
    {
        have_regex = true;
        cf_reader->set_files_regex(opt_vals["wind_files"].as<std::string>());
        wind_reader = cf_reader;
    }
    else if (have_regex)
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<std::string>());
        wind_reader = cf_reader;
    }

    if (opt_vals.count("track_file_out"))
        track_writer->set_file_name(opt_vals["track_file_out"].as<std::string>());

    p_teca_algorithm track_input;
    if (opt_vals.count("track_mask"))
    {
        track_filter->set_input_connection(track_reader->get_output_port());
        track_filter->set_mask_expression(opt_vals["track_mask"].as<std::string>());
        track_input = track_filter;
    }
    else
    {
        track_input = track_reader;
    }

    if (opt_vals.count("wind_u_var"))
        wind_radii->set_wind_u_variable(opt_vals["wind_u_var"].as<std::string>());

    if (opt_vals.count("wind_v_var"))
        wind_radii->set_wind_v_variable(opt_vals["wind_v_var"].as<std::string>());

    if (opt_vals.count("n_radial_bins"))
        wind_radii->set_number_of_radial_bins(opt_vals["n_radial_bins"].as<int>());

    if (opt_vals.count("profile_type"))
    {
        std::string profile_type = opt_vals["profile_type"].as<std::string>();
        if (profile_type == "avg")
        {
            wind_radii->set_profile_type(1);
        }
        else if (profile_type == "max")
        {
            wind_radii->set_profile_type(0);
        }
        else
        {
            TECA_ERROR("invalid profile_type " << profile_type)
            return -1;
        }
    }

    if (opt_vals.count("search_radius"))
        wind_radii->set_search_radius(opt_vals["search_radius"].as<double>());

    if (opt_vals.count("first_track"))
        map_reduce->set_start_index(opt_vals["first_track"].as<long>());

    if (opt_vals.count("last_track"))
        map_reduce->set_end_index(opt_vals["last_track"].as<long>());

    if (opt_vals.count("n_threads"))
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        map_reduce->set_thread_pool_size(-1);

    // some minimal check for missing options
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_ERROR("Extacly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader (HighResMIP datasets) "
                "and --input_regex to activate the cf_reader (CAM like datasets)")
        }
        return -1;
    }

    // connect the pipeline
    wind_coords->set_input_connection(wind_reader->get_output_port());
    wind_radii->set_input_connection(0, track_input->get_output_port());
    wind_radii->set_input_connection(1, wind_coords->get_output_port());
    map_reduce->set_input_connection(wind_radii->get_output_port());
    sort->set_input_connection(map_reduce->get_output_port());
    track_writer->set_input_connection(sort->get_output_port());

    // run the pipeline
    track_writer->update();

    return 0;
}
