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

    // initialize command line options description
    // set up some common options to simplify use for most
    // common scenarios
    options_description opt_defs(
        "teca_cartesian_mesh_diff an application that compares two datasets.\n\n"
        "Command line options", 120, -1
        );
    opt_defs.add_options()
        ("reference_dataset", value<std::string>()->required(),
            "cf_reader regex identifying the reference dataset")

        ("test_dataset", value<std::string>()->required(),
            "cf_reader regex identifying the test dataset")

        ("arrays", value<std::vector<std::string>>()->multitoken()->required(),
            "a list of arrays to compare")

        ("test_tolerance", value<double>()->default_value(1e-6),
            "max allowable relative difference in array values (1e-6)")

        ("start_index", value<long>()->default_value(0),
            "first time step to process (0)")

        ("end_index", value<long>()->default_value(-1),
            "last time step to process (-1)")

        ("verbose", "enable extra terminal output")
        ("help", "display the basic options help")
        ;

    // parse the command line
    variables_map opt_vals;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(opt_defs).run(),
            opt_vals);

        if (mpi_man.get_comm_rank() == 0)
        {
            if (opt_vals.count("help"))
            {
                std::cerr << std::endl
                    << "usage: teca_cartesian_mesh_diff [options]" << std::endl
                    << std::endl
                    << opt_defs << std::endl
                    << std::endl;
                return -1;
            }
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        if (mpi_man.get_comm_rank() == 0)
        {
           TECA_ERROR("Error parsing command line options. See --help "
               "for a list of supported options. " << e.what())
        }
        return -1;
    }

    std::string ref_file = opt_vals["reference_dataset"].as<std::string>();
    std::string test_file = opt_vals["test_dataset"].as<std::string>();
    std::vector<std::string> arrays = opt_vals["arrays"].as<std::vector<std::string>>();
    double test_tol = opt_vals["test_tolerance"].as<double>();
    long start_index = opt_vals["start_index"].as<long>();
    long end_index = opt_vals["end_index"].as<long>();


    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_start_index(start_index);
    exec->set_end_index(end_index);
    exec->set_arrays(arrays);
    if (opt_vals.count("verbose"))
        exec->set_verbose(1);

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test)
    {
        TECA_STATUS("Running the test")

        p_teca_algorithm ref_reader = teca_cartesian_mesh_reader_factory::New(ref_file);
        p_teca_algorithm test_reader = teca_cartesian_mesh_reader_factory::New(test_file);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, ref_reader->get_output_port());
        diff->set_input_connection(1, test_reader->get_output_port());
        diff->set_tolerance(test_tol);
        diff->set_executive(exec);

        diff->update();
    }
    else
    {
        TECA_STATUS("Writing the baseline")

        p_teca_algorithm reader = teca_cartesian_mesh_reader_factory::New(test_file);

        p_teca_algorithm writer = teca_cartesian_mesh_writer_factory::New(ref_file);
        writer->set_input_connection(reader->get_output_port());
        writer->set_executive(exec);

        writer->update();
    }

    return 0;
}
