#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_temporal_average.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_ar_detect.h"
#include "teca_table_sort.h"
#include "teca_table_calendar.h"
#include "teca_table_reduce.h"
#include "teca_table_writer.h"
#include "teca_time_step_executive.h"
#include "teca_mpi_manager.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

using namespace std;

using seconds_t =
    std::chrono::duration<double, std::chrono::seconds::period>;

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
using boost::program_options::value;
#endif

#define TECA_TIME
#if defined TECA_TIME
#include <sys/time.h>
#endif

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    std::chrono::high_resolution_clock::time_point t0, t1;
    if (mpi_man.get_comm_rank() == 0)
        t0 = std::chrono::high_resolution_clock::now();

    // create pipeline objects here so that they
    // can be initialized from the command line
    p_teca_cf_reader water_vapor_reader = teca_cf_reader::New();
    p_teca_temporal_average water_vapor_average = teca_temporal_average::New();
    p_teca_cf_reader land_sea_mask_reader = teca_cf_reader::New();
    p_teca_cartesian_mesh_regrid land_sea_mask_regrid = teca_cartesian_mesh_regrid::New();
    p_teca_ar_detect ar_detect = teca_ar_detect::New();
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    p_teca_table_writer results_writer = teca_table_writer::New();

    // optional pipeline components
    bool use_time_average = false;
    bool use_land_sea_mask = false;

    // initialize command line options description
#if defined(TECA_HAS_BOOST)
    // set up some common options to simplify use for most
    // common scenarios
    options_description basic_opt_defs(
        "Basic usage:\n\n"
        "The following options are the most commonly used. However, there\n"
        "are numerous advanced control paramters exposed by the internal\n"
        "stages of the pipeline. Information on advanced options can be\n"
        "displayed using --advanced_help\n\n"
        "Basic command line options"
        );
    basic_opt_defs.add_options()
        ("water_vapor_file", value<string>(), "path of file containing water vapor data")
        ("water_vapor_file_regex", value<string>(), "regex matching files containing water vapor data")
        ("water_vapor_var", value<string>(), "name of water vapor variable")
        ("water_vapor_avg", value<int>(), "average water vapor variable over n time steps")
        ("land_sea_mask_file", value<string>(), "path of file containing land-sea mask")
        ("land_sea_mask_var", value<string>(), "name of land sea mask variable")
        ("first_step", value<long>(), "first time step to process")
        ("last_step", value<long>(), "last time step to process")
        ("n_threads", value<int>(), "thread pool size. default is 1. -1 for all")
        ("results_file", value<string>(), "path to write table of detected rivers to")
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
        "AR Detect pipeline:\n\n"
        "water_vapor_reader\n"
        " \\\n"
        " *water_vapor_average -- *land_sea_mask_regrid\n"
        "                         / \\\n"
        "    *land_sea_mask_reader   ar_detect\n"
        "                             \\\n"
        "                              map_reduce\n"
        "                               \\\n"
        "                                results_writer\n\n"
        "* denotes an optional stage.\n\n"
        "Advanced command line options"
        );
    water_vapor_reader->get_properties_description("water_vapor_reader", advanced_opt_defs);
    water_vapor_average->get_properties_description("water_vapor_average", advanced_opt_defs);
    land_sea_mask_reader->get_properties_description("land_sea_mask_reader", advanced_opt_defs);
    land_sea_mask_regrid->get_properties_description("land_sea_mask_regrid", advanced_opt_defs);
    ar_detect->get_properties_description("ar_detect", advanced_opt_defs);
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);
    results_writer->get_properties_description("results_writer", advanced_opt_defs);

    options_description all_opt_defs;
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(all_opt_defs).run(),
            opt_vals);

        if (rank == 0)
        {
            if (opt_vals.count("help"))
            {
                cerr << endl
                    << "usage: teca_ar_detect [options]" << endl
                    << endl
                    << basic_opt_defs << endl
                    << endl;
                return -1;
            }
            if (opt_vals.count("advanced_help"))
            {
                cerr << endl
                    << "usage: teca_ar_detect [options]" << endl
                    << endl
                    << advanced_opt_defs << endl
                    << endl;
                return -1;
            }

            if (opt_vals.count("full_help"))
            {
                cerr << endl
                    << "usage: teca_ar_detect [options]" << endl
                    << endl
                    << all_opt_defs << endl
                    << endl;
                return -1;
            }
        }

        boost::program_options::notify(opt_vals);
    }
    catch(std::exception &e)
    {
        cerr << endl
            << "Error parsing command line options. See --help "
               "for a list of supported options." << endl
            << e.what() << endl;
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, thus basic options
    // override them
    water_vapor_reader->set_properties("water_vapor_reader", opt_vals);
    water_vapor_average->set_properties("water_vapor_average", opt_vals);
    land_sea_mask_reader->set_properties("land_sea_mask_reader", opt_vals);
    land_sea_mask_regrid->set_properties("land_sea_mask_regrid", opt_vals);
    ar_detect->set_properties("ar_detect", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    results_writer->set_properties("results_writer", opt_vals);

    // process the basic options
    if (opt_vals.count("water_vapor_file"))
        water_vapor_reader->set_file_name(opt_vals["water_vapor_file"].as<string>());

    if (opt_vals.count("water_vapor_file_regex"))
        water_vapor_reader->set_files_regex(opt_vals["water_vapor_file_regex"].as<string>());

    if (opt_vals.count("water_vapor_var"))
        ar_detect->set_water_vapor_variable(opt_vals["water_vapor_var"].as<string>());

    if (opt_vals.count("water_vapor_avg"))
        water_vapor_average->set_filter_width(opt_vals["water_vapor_avg"].as<int>());

    if (opt_vals.count("land_sea_mask_file"))
        land_sea_mask_reader->set_file_name(opt_vals["land_sea_mask_file"].as<string>());

    if (opt_vals.count("land_sea_mask_var"))
    {
        string var = opt_vals["land_sea_mask_var"].as<string>();
        ar_detect->set_land_sea_mask_variable(var);
        land_sea_mask_regrid->add_source_array(var);
    }

    if (opt_vals.count("first_step"))
        map_reduce->set_first_step(opt_vals["first_step"].as<long>());

    if (opt_vals.count("last_step"))
        map_reduce->set_last_step(opt_vals["last_step"].as<long>());

    if (opt_vals.count("n_threads"))
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    if (opt_vals.count("results_file"))
        results_writer->set_file_name(opt_vals["results_file"].as<string>());

    // detect request for optional pipeline stages
    if (!land_sea_mask_reader->get_file_name().empty()
        || !land_sea_mask_reader->get_files_regex().empty())
        use_land_sea_mask = true;

    if (opt_vals.count("water_vapor_average::filter_width")
        || opt_vals.count("water_vapor_avg"))
        use_time_average = true;

    // minimal check for missing options
    if (water_vapor_reader->get_file_name().empty()
        && water_vapor_reader->get_files_regex().empty())
    {
        cerr << endl
            << "Error: missing file name or regex for water vapor dataset. "
               "See --help for a list of command line options."
            << endl;
        return -1;
    }

    if (use_land_sea_mask
        && ((land_sea_mask_reader->get_file_name().empty()
        && land_sea_mask_reader->get_files_regex().empty())
        || land_sea_mask_regrid->get_source_arrays().empty()))
    {
        cerr << endl
            << "Error: missing file name, regex, or variable name "
               "for land sea mask dataset. See --help for a list of "
               "command line options."
            << endl;
        return -1;
    }

    if (results_writer->get_file_name().empty())
    {
        cerr << endl
            << "Error: missing results file. "
               "See --help for a list of command line options."
            << endl;
        return -1;
    }
#endif

    // build the pipeline
    p_teca_algorithm alg_0 = water_vapor_reader;
    if (use_time_average)
    {
        // add optional stage for temporal average
        water_vapor_average->set_input_connection(water_vapor_reader->get_output_port());
        alg_0 = water_vapor_average;
    }

    p_teca_algorithm alg_1 = alg_0;
    if (use_land_sea_mask)
    {
        // add optional land sea mask reader and regrid stages
        land_sea_mask_reader->set_t_axis_variable("");

        land_sea_mask_regrid->set_input_connection(0, alg_0->get_output_port());
        land_sea_mask_regrid->set_input_connection(1, land_sea_mask_reader->get_output_port());
        alg_1 = land_sea_mask_regrid;
    }

    // add ar detector stage
    ar_detect->set_input_connection(alg_1->get_output_port());

    // add map reduce stage
    map_reduce->set_input_connection(ar_detect->get_output_port());

    // sort and compute dates
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("time_step");

    p_teca_table_calendar calendar = teca_table_calendar::New();
    calendar->set_input_connection(sort->get_output_port());

    // writer
    results_writer->set_input_connection(calendar->get_output_port());

    // run the pipeline
    results_writer->update();

    if (mpi_man.get_comm_rank() == 0)
    {
        t1 = std::chrono::high_resolution_clock::now();
        seconds_t dt(t1 - t0);
        TECA_STATUS("teca_ar_detect run_time=" << dt.count() << " sec")
    }

    return 0;
}
