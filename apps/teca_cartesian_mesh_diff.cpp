#include "teca_config.h"
#include "teca_common.h"
#include "teca_algorithm.h"
#include "teca_cartesian_mesh_reader_factory.h"
#include "teca_cartesian_mesh_writer_factory.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
#include <boost/program_options.hpp>
using boost::program_options::value;

#include <string>
#include <iostream>
#include <vector>


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // are we doing the test or updating the baseline?
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);

    // grab file names first so we can construct an instance
    // then we can use the instance to get advanced options.
    std::string ref_file;
    if (teca_system_util::get_command_line_option(
        argc, argv, "--reference_dataset", 1, ref_file))
        return -1;

    std::string test_file;
    if (teca_system_util::get_command_line_option(
        argc, argv, "--test_dataset", 1, test_file))
        return -1;

    p_teca_algorithm test_reader = teca_cartesian_mesh_reader_factory::New(test_file);
    if (!test_reader)
    {
        TECA_FATAL_ERROR("the test file format was not recognized from \""
            << test_file << "\"")
        return -1;
    }

    p_teca_algorithm ref_reader;
    p_teca_algorithm ref_writer;

    if (do_test)
        ref_reader = teca_cartesian_mesh_reader_factory::New(ref_file);
    else
        ref_writer = teca_cartesian_mesh_writer_factory::New(ref_file);

    if (!ref_reader && !ref_writer)
    {
        TECA_FATAL_ERROR("the refence file format was not recognized from \""
            << ref_file << "\"")
        return -1;
    }

    p_teca_dataset_diff diff = teca_dataset_diff::New();

    // initialize command line options description
    // set up some common options to simplify use for most
    // common scenarios
    int help_width = 100;
    options_description basic_opt_defs(
        "teca_cartesian_mesh_diff an application that compares two datasets.\n\n"
        "Command line options", help_width, help_width - 4
        );
    basic_opt_defs.add_options()
        ("reference_dataset", value<std::string>()->required(),
            "cf_reader regex identifying the reference dataset")

        ("test_dataset", value<std::string>()->required(),
            "cf_reader regex identifying the test dataset")

        ("arrays", value<std::vector<std::string>>()->multitoken()->required(),
            "a list of arrays to compare")

        ("relative_tolerance", value<double>()->default_value(-1.0),
            "max allowable relative difference in array values")

        ("absolute_tolerance", value<double>()->default_value(-1.0),
            "max allowable relative difference in array values")

        ("start_index", value<long>()->default_value(0),
            "first time step to process (0)")

        ("end_index", value<long>()->default_value(-1),
            "last time step to process (-1)")

        ("verbose", "enable extra terminal output")
        ("help", "display the basic options help")
        ("full_help", "display all options help information")
        ("advanced_help", "display the advanced options help")
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
    test_reader->get_properties_description("test_reader", advanced_opt_defs);
    if (do_test)
    {
        ref_reader->get_properties_description("ref_reader", advanced_opt_defs);
        diff->get_properties_description("diff", advanced_opt_defs);
    }
    else
    {
        ref_writer->get_properties_description("ref_writer", advanced_opt_defs);
    }

    // package basic and advanced options for display
    options_description all_opt_defs(help_width, help_width - 4);
    all_opt_defs.add(basic_opt_defs).add(advanced_opt_defs);

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
                .style(boost::program_options::command_line_style::unix_style ^
                       boost::program_options::command_line_style::allow_short)
                .options(all_opt_defs)
                .run(),
                opt_vals);

        if (mpi_man.get_comm_rank() == 0)
        {
            if (opt_vals.count("help"))
            {
                std::cerr << std::endl
                    << "usage: teca_cartesian_mesh_diff [options]" << std::endl
                    << std::endl
                    << basic_opt_defs << std::endl
                    << std::endl;
                return 0;
            }

            if (opt_vals.count("advanced_help"))
            {
                std::cerr << std::endl
                    << "usage: teca_cartesian_mesh_diff [options]" << std::endl
                    << std::endl
                    << advanced_opt_defs << std::endl
                    << std::endl;
                return 0;
            }

            if (opt_vals.count("full_help"))
            {
                std::cerr << std::endl
                    << "usage: teca_cartesian_mesh_diff [options]" << std::endl
                    << std::endl
                    << all_opt_defs << std::endl
                    << std::endl;
                return 0;
            }
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        if (mpi_man.get_comm_rank() == 0)
        {
           TECA_FATAL_ERROR("Error parsing command line options. See --help "
               "for a list of supported options. " << e.what())
        }
        return -1;
    }

    // pass command line arguments into the pipeline objects
    // advanced options are processed first, so that the basic
    // options will override them
    test_reader->set_properties("test_reader", opt_vals);
    if (do_test)
    {
        ref_reader->set_properties("ref_reader", opt_vals);
        diff->set_properties("diff", opt_vals);
    }
    else
    {
        ref_writer->set_properties("ref_writer", opt_vals);
    }

    if (!opt_vals["relative_tolerance"].defaulted())
    {
        diff->set_relative_tolerance(opt_vals["relative_tolerance"].as<double>());
    }

    if (!opt_vals["absolute_tolerance"].defaulted())
    {
        diff->set_absolute_tolerance(opt_vals["absolute_tolerance"].as<double>());
    }

    std::vector<std::string> arrays = opt_vals["arrays"].as<std::vector<std::string>>();
    long start_index = opt_vals["start_index"].as<long>();
    long end_index = opt_vals["end_index"].as<long>();

    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_start_index(start_index);
    exec->set_end_index(end_index);
    exec->set_arrays(arrays);

    bool verbose = opt_vals.count("verbose");
    if (verbose)
    {
        exec->set_verbose(1);
        diff->set_verbose(1);
    }
    else
    {
        exec->set_verbose(0);
        diff->set_verbose(0);
    }

    if (do_test)
    {
        TECA_STATUS("Running the test")

        diff->set_input_connection(0, ref_reader->get_output_port());
        diff->set_input_connection(1, test_reader->get_output_port());
        diff->set_executive(exec);

        diff->update();
    }
    else
    {
        TECA_STATUS("Writing the baseline")

        ref_writer->set_input_connection(test_reader->get_output_port());
        ref_writer->set_executive(exec);

        ref_writer->update();
    }

    return 0;
}
