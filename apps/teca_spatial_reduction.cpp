#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_spatial_reduction.h"
#include "teca_app_util.h"
#include "teca_mpi_manager.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_writer.h"

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
        "using a predefined spatial reduction operator", help_width, help_width - 4);

    basic_opt_defs.add_options()
        ("input_file", value<std::string>(), "\nA teca_multi_cf_reader configuration file"
            " identifying the set of NetCDF CF2 files to process. When present data is"
            " read using the teca_multi_cf_reader. Use one of either --input_file or"
            " --input_regex.\n")

        ("input_regex", value<std::string>(), "\nA teca_cf_reader regex identifying the"
            " set of NetCDF CF2 files to process. When present data is read using the"
            " teca_cf_reader. Use one of either --input_file or --input_regex.\n")

        ("operator", value<std::string>()->default_value("average"),
            "\nReduction operator to use. One of summation, minimum, maximum, or average.\n")

        ("point_arrays", value<std::vector<std::string>>()->multitoken(),
            "\nA list of point centered arrays to process.\n")

        ("fill_value", value<double>(), "\nA value that identifies missing or invalid data."
            " Specifying the fill value on the command line"
            " overrides array specific fill values stored in the file.\n")

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

        ("output_file", value<string>()->default_value("output.csv"),
            "\nfile path to write the storm candidates to. The extension determines"
            " the file format. May be one of `.nc`, `.csv`, or `.bin`\n")

        ("land_weights", value<string>()->default_value(""),
            "\nname of variable with land weights\n")

        ("land_weights_norm", value<double>(),
            "\nland weights norm\n")

        ("first_step", value<long>()->default_value(0),
            "\nfirst time step to process\n")

        ("last_step", value<long>()->default_value(-1),
            "\nlast time step to process\n")

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
    p_teca_spatial_reduction red = teca_spatial_reduction::New();
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    p_teca_table_sort sort = teca_table_sort::New();
    p_teca_table_writer table_writer = teca_table_writer::New();

    cf_reader->get_properties_description("cf_reader", advanced_opt_defs);
    mcf_reader->get_properties_description("multi_cf_reader", advanced_opt_defs);
    vv_mask->get_properties_description("valid_value_mask", advanced_opt_defs);
    unpack->get_properties_description("unpack_data", advanced_opt_defs);
    red->get_properties_description("spatial_reduction", advanced_opt_defs);
    map_reduce->get_properties_description("map_reduce", advanced_opt_defs);
    sort->get_properties_description("sort", advanced_opt_defs);
    table_writer->get_properties_description("table_writer", advanced_opt_defs);

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
    red->set_properties("spatial_reduction", opt_vals);
    map_reduce->set_properties("map_reduce", opt_vals);
    sort->set_properties("sort", opt_vals);
    table_writer->set_properties("table_writer", opt_vals);

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
        std::cerr << opt_vals["input_regex"].as<string>() << std::endl;
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

    if (opt_vals["land_weights"].as<string>() == "")
        TECA_FATAL_ERROR("Missing name of variable with land weights")
    red->set_land_weights(opt_vals["land_weights"].as<string>());

    if (opt_vals.count("land_weights_norm"))
        red->set_land_weights_norm(opt_vals["land_weights_norm"].as<double>());


    if (!opt_vals["first_step"].defaulted())
        map_reduce->set_start_index(opt_vals["first_step"].as<long>());

    if (!opt_vals["last_step"].defaulted())
        map_reduce->set_end_index(opt_vals["last_step"].as<long>());

    if (!opt_vals["n_threads"].defaulted())
        map_reduce->set_thread_pool_size(opt_vals["n_threads"].as<int>());
    else
        map_reduce->set_thread_pool_size(-1);
    map_reduce->set_input_connection(red->get_output_port());

    sort->set_index_column("step");
    sort->set_input_connection(map_reduce->get_output_port());

    table_writer->set_output_format_auto();
    table_writer->set_file_name(opt_vals["output_file"].as<string>());
    table_writer->set_input_connection(sort->get_output_port());

    if (mpi_man.get_comm_rank() == 0 && opt_vals["verbose"].as<int>())
    {
       std::cerr << "running on " << mpi_man.get_comm_size() << " ranks" << std::endl;
       std::cerr << "operator=" << opt_vals["operator"].as<string>() << std::endl;
       std::cerr << "point_arrays=" << opt_vals["point_arrays"].as<std::vector<std::string>>()<< std::endl;
    }

    table_writer->update();

    return 0;
}
