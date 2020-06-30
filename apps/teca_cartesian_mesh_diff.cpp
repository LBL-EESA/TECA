#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_table_sort.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_mpi.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc < 4)
    {
        if (rank == 0)
        {
            cerr << endl
                << "Usage:" << endl
                << "teca_cartesian_mesh_diff [test file type] [test file] "
                   "[reference file]" << endl
                 << endl;
        }
        return -1;
    }

    // parse command line
    unsigned int t_file_type = atoi(argv[1]);
    std::string t_file = argv[2];
    std::string ref_file = argv[3];

    if (!teca_file_util::file_exists(t_file.c_str()))
    {
        TECA_ERROR("output file doesn't exist");
        return -1;
    }

    if (!teca_file_util::file_exists(ref_file.c_str()))
    {
        TECA_ERROR("no reference file to compare to");
        return -1;
    }

    p_teca_cf_reader cf_reader = teca_cf_reader::New();

    p_teca_cartesian_mesh_reader m_reader = teca_cartesian_mesh_reader::New();

    p_teca_dataset_diff diff = teca_dataset_diff::New();

    p_teca_cartesian_mesh_reader ref_reader = teca_cartesian_mesh_reader::New();
    ref_reader->set_file_name(ref_file);

    diff->set_input_connection(0, ref_reader->get_output_port());

    switch (t_file_type)
    {
        // 0 -> nc format
        case 0:
        {
            cf_reader->append_file_name(t_file);
            diff->set_input_connection(1, cf_reader->get_output_port());
            break;
        }
        // 1 -> bin format
        case 1:
        {
            m_reader->set_file_name(t_file);
            diff->set_input_connection(1, m_reader->get_output_port());
            break;
        }
        default:
        {
            TECA_ERROR("input file format type" << t_file_type
                << " is not supported");
            return -1;
        }
    }
    

    diff->update();

    return 0;
}
