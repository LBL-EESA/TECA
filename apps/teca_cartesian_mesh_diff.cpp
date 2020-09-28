#include "teca_config.h"
#include "teca_common.h"
#include "teca_algorithm.h"
#include "teca_cartesian_mesh_reader_factory.h"
#include "teca_cartesian_mesh_writer_factory.h"
#include "teca_dataset_diff.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"

#include <string>
#include <iostream>

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc != 3)
    {
        if (rank == 0)
        {
            std::cerr << std::endl << "Usage:" << std::endl
                << "teca_cartesian_mesh_diff [reference file] [test file]"
                << std::endl << std::endl;
        }
        return -1;
    }

    std::string test_file = argv[1];
    std::string ref_file = argv[2];

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

        diff->update();
    }
    else
    {
        TECA_STATUS("Writing the baseline")

        p_teca_algorithm reader = teca_cartesian_mesh_reader_factory::New(test_file);

        p_teca_algorithm writer = teca_cartesian_mesh_writer_factory::New(ref_file);
        writer->set_input_connection(reader->get_output_port());

        writer->update();
    }

    return 0;
}
