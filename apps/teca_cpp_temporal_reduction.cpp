#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_temporal_reduction.h"
#include "teca_app_util.h"
#include "teca_mpi_manager.h"

#include <vector>
#include <string>
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;

using boost::program_options::value;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // initialize mpi
    teca_mpi_manager mpi_man(argc, argv);

    int help_width = 100;
    options_description basic_opt_defs(
        "Reduce the time axis of a NetcCDF CF2 dataset"
        "using a predefined interval and reduction operator", help_width, help_width - 4);

    basic_opt_defs.add_options()
        ("input_file", value<std::string>(), "\nA teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\nA teca_cf_reader regex identifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("interval", value<std::string>()->default_value("monthly"),
            "\nInterval to reduce the time axis to. One of daily, monthly, seasonal,"
            " yearly, or n_steps. For the n_steps option use --number_of_steps.\n")

        ("number_of_steps", value<long>()->default_value(0),
            "\nThe desired number of steps when --interval n_steps is specified.\n")

        ("operator", value<std::string>()->default_value("average"),
            "\nReduction operator to use. One of summation, minimum, maximum, or average.\n")

        ("point_arrays", value<std::vector<std::string>>()->multitoken(),
            "\nA list of point centered arrays to process.\n")

        ("fill_value", value<double>(), "\nA value that identifies missing or invalid data."
            " Specifying the fill value on the command line"
            " overrides array specific fill values stored in the file.\n")

        ("output_file", value<std::string>(),
            "\nA path and file name pattern for the output NetCDF files. %t% is replaced with a"
            " human readable date and time corresponding to the time of the first time step in"
            " the file. Use --cf_writer::date_format to change the formatting.\n")

        ("file_layout", value<std::string>()->default_value("yearly"),
            "\nSelects the size and layout of the set of output files. May be one of"
            " number_of_steps, daily, monthly, seasonal, or yearly. Files are structured"
            " such that each file contains one of the selected interval. For the number_of_steps"
            " option use --steps_per_file.\n")

        ("steps_per_file", value<long>()->default_value(128),
            "\nThe number of time steps per output file when --file_layout number_of_steps is specified.\n")

        ("x_axis_variable", value<std::string>()->default_value("lon"),
            "\nName of the variable to use for x-coordinates.\n")

        ("y_axis_variable", value<std::string>()->default_value("lat"),
            "\nName of the variable to use for y-coordinates.\n")

        ("z_axis_variable", value<std::string>()->default_value(""),
            "\nName of z coordinate variable. When processing 3D set this to"
            " the variable containing vertical coordinates. When empty the"
            " data will be treated as 2D.\n")

        ("t_axis_variable", value<std::string>()->default_value("time"),
            "\nName of the variable to use for t-coordinates.\n")

        ("n_threads", value<int>()->default_value(-1), "\nSets the thread pool"
            " size on each MPI rank. When the default value of -1 is used TECA"
            " will coordinate the thread pools across ranks such each thread"
            " is bound to a unique physical core.\n")

        ("steps_per_request", value<int>()->default_value(1), "\nSets the number"
         " of time steps per request\n")

        ("spatial_partitioning", "\nActivates the spatial partitioning engine.\n")

        ("spatial_partitions", value<int>()->default_value(0), "\nSets the number of spatial partitions."
            " Use zero for automatic partitioning and 1 for no partitioning\n")

        ("verbose", value<int>()->default_value(0), "\nUse 1 to enable verbose mode,"
            " otherwise 0.\n")
        ("help", "\ndisplays documentation for application specific command line options\n")
        ("advanced_help", "\ndisplays documentation for algorithm specific command line options\n")
        ("full_help", "\ndisplays both basic and advanced documentation together\n")
        ;

    options_description advanced_opt_defs(
        "Advanced usage:\n\n"
        "The following list contains the full set options giving one full\n"
        "control over all runtime modifiable parameters. The basic options\n"
        "(see" "--help) map to these, and will override them if both are\n"
        "specified.\n\n"
        "Advanced command line options", help_width, help_width - 4
        );

    p_teca_multi_cf_reader mcf_reader = teca_multi_cf_reader::New();
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    p_teca_valid_value_mask vv_mask = teca_valid_value_mask::New();
    p_teca_unpack_data unpack = teca_unpack_data::New();
    p_teca_cpp_temporal_reduction red = teca_cpp_temporal_reduction::New();
    p_teca_cf_writer cf_writer = teca_cf_writer::New();

    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);
    mcf_reader->get_properties_description("multi_cf_reader", advanced_opt_defs);
    vv_mask->get_properties_description("valid_value_mask", advanced_opt_defs);
    unpack->get_properties_description("unpack_data", advanced_opt_defs);
    red->get_properties_description("cpp_temporal_reduction", advanced_opt_defs);
    cf_writer->get_properties_description("cf_writer", advanced_opt_defs);

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

    mcf_reader->set_properties("multi_cf_reader", opt_vals);
    cf_reader->set_properties("cf_reader", opt_vals);
    vv_mask->set_properties("valid_value_mask", opt_vals);
    unpack->set_properties("unpack_data", opt_vals);
    red->set_properties("cpp_temporal_reduction", opt_vals);
    cf_writer->set_properties("cf_writer", opt_vals);

    bool have_file = opt_vals.count("input_file");
    bool have_regex = opt_vals.count("input_regex");
    if ((have_file && have_regex) || !(have_file || have_regex))
    {
        if (mpi_man.get_comm_rank() == 0)
        {
            TECA_FATAL_ERROR("Exactly one of --input_file or --input_regex can be specified. "
                "Use --input_file to activate the multi_cf_reader "
                "and --input_regex to activate the cf_reader")
        }
        return -1;
    }

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

    if (!opt_vals["z_axis_variable"].defaulted())
    {
        cf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
        mcf_reader->set_z_axis_variable(opt_vals["z_axis_variable"].as<std::string>());
    }

    if (!opt_vals["t_axis_variable"].defaulted())
    {
        cf_reader->set_t_axis_variable(opt_vals["t_axis_variable"].as<std::string>());
        mcf_reader->set_t_axis_variable(opt_vals["t_axis_variable"].as<std::string>());
    }

    if (opt_vals.count("input_file"))
    {
        mcf_reader->set_input_file(opt_vals["input_file"].as<string>());
        vv_mask->set_input_connection(mcf_reader->get_output_port());
    }
    else if (opt_vals.count("input_regex"))
    {
        cf_reader->set_files_regex(opt_vals["input_regex"].as<string>());
        vv_mask->set_input_connection(cf_reader->get_output_port());
    }


    if (opt_vals["verbose"].as<int>() > 1)
       vv_mask->set_verbose(1);
    else
       vv_mask->set_verbose(0);

    unpack->set_input_connection(vv_mask->get_output_port());
    unpack->set_verbose(opt_vals["verbose"].as<int>());

    red->set_input_connection(unpack->get_output_port());
    red->set_stream_size(2);
    red->set_verbose(opt_vals["verbose"].as<int>());
    red->set_thread_pool_size(opt_vals["n_threads"].as<int>());

    if (!opt_vals["interval"].defaulted())
    {
        red->set_interval(opt_vals["interval"].as<string>());
        if (opt_vals["interval"].as<string>() == "n_steps" &&
           !opt_vals["number_of_steps"].defaulted())
        {
            red->set_number_of_steps(opt_vals["number_of_steps"].as<long>());
        }
    }

    if (!opt_vals["operator"].defaulted())
    {
        red->set_operation(opt_vals["operator"].as<string>());
    }

    if (opt_vals.count("point_arrays"))
    {
        red->set_point_arrays(opt_vals["point_arrays"].as<std::vector<std::string>>());
    }

    if (opt_vals.count("fill_value"))
    {
        red->set_fill_value(opt_vals["fill_value"].as<double>());
    }

    if (!opt_vals["steps_per_request"].defaulted())
    {
        red->set_steps_per_request(opt_vals["steps_per_request"].as<int>());
    }

    cf_writer->set_input_connection(red->get_output_port());
    cf_writer->set_stream_size(2);
    cf_writer->set_verbose(opt_vals["verbose"].as<int>());
    cf_writer->set_thread_pool_size(1);

    if (opt_vals.count("output_file"))
    {
       cf_writer->set_file_name(opt_vals["output_file"].as<string>());
    }

    if (!opt_vals["steps_per_file"].defaulted())
    {
        cf_writer->set_steps_per_file(opt_vals["steps_per_file"].as<long>());
    }

    if (!opt_vals["file_layout"].defaulted() &&
        cf_writer->set_layout(opt_vals["file_layout"].as<std::string>()))
    {
        TECA_FATAL_ERROR("An invalid file layout was provided \""
            << opt_vals["file_layout"].as<std::string>() << "\"")
        return -1;
    }

    if (!opt_vals.count("point_arrays"))
    {
        TECA_FATAL_ERROR("The following command line arguments are required: --point_arrays")
        return -1;
    }

    cf_writer->set_point_arrays(opt_vals["point_arrays"].as<std::vector<std::string>>());

    cf_writer->set_index_executive_compatability(1);
    cf_writer->set_number_of_spatial_partitions(opt_vals["spatial_partitions"].as<int>());
    if (opt_vals.count("spatial_partitioning"))
    {
        cf_writer->set_partitioner(teca_cf_writer::space_time);
    }
    else
    {
        cf_writer->set_partitioner(teca_cf_writer::temporal);
    }

    if (mpi_man.get_comm_rank() == 0 && opt_vals["verbose"].as<int>())
    {
       std::cerr << "running on " << mpi_man.get_comm_size() << " ranks" << std::endl;
       std::cerr << "file_layout=" << opt_vals["file_layout"].as<string>() << std::endl;
       std::cerr << "steps_per_file=" << opt_vals["steps_per_file"].as<long>() << std::endl;
       std::cerr << "interval=" << opt_vals["interval"].as<string>() << std::endl;
       std::cerr << "operator=" << opt_vals["operator"].as<string>() << std::endl;
       std::cerr << "point_arrays=" << opt_vals["point_arrays"].as<std::vector<std::string>>()<< std::endl;
    }

    cf_writer->update();

    return 0;
}
