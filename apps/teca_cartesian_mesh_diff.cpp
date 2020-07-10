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

    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl
                << "Usage:" << endl
                << "teca_cartesian_mesh_diff [test file] "
                   "[reference file]" << endl
                << endl;
        }
        return -1;
    }

    // parse command line
    std::string t_file = argv[1];
    std::string ref_file = argv[2];

    cerr << "t_file: " << t_file << endl;
    cerr << "ref_file: " << ref_file << endl;

    if (!teca_file_util::file_exists(ref_file.c_str()))
    {
        TECA_ERROR("no reference file to compare to");
        return -1;
    }

    p_teca_cf_reader cfr_t_file = teca_cf_reader::New();
    p_teca_cf_reader cfr_ref_file = teca_cf_reader::New();

    p_teca_cartesian_mesh_reader mr_t_file = teca_cartesian_mesh_reader::New();
    p_teca_cartesian_mesh_reader mr_ref_file = teca_cartesian_mesh_reader::New();

    p_teca_dataset_diff diff = teca_dataset_diff::New();

    std::string t_file_type = teca_file_util::extension(t_file);
    std::string ref_file_type = teca_file_util::extension(ref_file);

    if (t_file_type == "nc" && ref_file_type == "nc")
    {
        cfr_ref_file->set_files_regex(ref_file);
        cfr_t_file->set_files_regex(t_file);

        diff->set_input_connection(0, cfr_ref_file->get_output_port());
        diff->set_input_connection(1, cfr_t_file->get_output_port());
    }
    else if (t_file_type == "nc" && ref_file_type == "bin")
    {
        mr_ref_file->set_file_name(ref_file);
        cfr_t_file->set_files_regex(t_file);

        diff->set_input_connection(0, mr_ref_file->get_output_port());
        diff->set_input_connection(1, cfr_t_file->get_output_port());
    }
    else if (t_file_type == "bin" && ref_file_type == "nc")
    {
        cfr_ref_file->set_files_regex(ref_file);
        mr_t_file->set_file_name(t_file);

        diff->set_input_connection(0, cfr_ref_file->get_output_port());
        diff->set_input_connection(1, mr_t_file->get_output_port());
    }
    else if (t_file_type == "bin" && ref_file_type == "bin")
    {
        mr_ref_file->set_file_name(ref_file);
        mr_t_file->set_file_name(t_file);

        diff->set_input_connection(0, mr_ref_file->get_output_port());
        diff->set_input_connection(1, mr_t_file->get_output_port());
    }
    else
    {
        TECA_ERROR("input or reference files format type - test file type: "
            << t_file_type << " or reference file type: " << ref_file_type
            << " is not supported");
        return -1;
    }

    diff->update();

    return 0;
}
