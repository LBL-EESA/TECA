#include "teca_config.h"
#include "teca_table_reader.h"
#include "teca_table_writer.h"
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

    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl
                << "Usage:" << endl
                << "teca_table_diff [test file] [reference file] "
                   "[column to sort by (optional)]" << endl
                 << endl;
        }
        return -1;
    }

    // parse command line
    std::string t_file = argv[1];
    std::string ref_file = argv[2];
    std::string column_name;
    if (argc == 4) column_name = argv[3];
    

    if (!teca_file_util::file_exists(t_file.c_str()))
    {
        TECA_ERROR("test file doesn't exist");
        return -1;
    }

    if (!teca_file_util::file_exists(ref_file.c_str()))
    {
        TECA_ERROR("no reference file to compare to");
        return -1;
    }

    p_teca_table_reader t_reader = teca_table_reader::New();
    t_reader->set_file_name(t_file);

    p_teca_table_reader ref_reader = teca_table_reader::New();
    ref_reader->set_file_name(ref_file);
    
    p_teca_table_sort sort = teca_table_sort::New();

    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, ref_reader->get_output_port());

    if (!column_name.empty())
    {
        sort->set_input_connection(t_reader->get_output_port());
        sort->set_index_column(column_name);

        diff->set_input_connection(1, sort->get_output_port());
    }
    else
    {
        diff->set_input_connection(1, t_reader->get_output_port());
    }

    diff->update();

    return 0;
}
