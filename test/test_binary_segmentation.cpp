#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_binary_segmentation.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_file_util.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_dataset_diff.h"
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

    if (argc != 6)
    {
        cerr << endl << "Usage error:" << endl
            << "test_binary_segmentation [input regex] [var] [low] [high] [output]"
            << endl << endl;
        return -1;
    }

    std::string in_files = argv[1];
    std::string var = argv[2];
    double low = atof(argv[3]);
    double high = atof(argv[4]);
    std::string baseline = argv[5];

    // build the pipeline
    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(in_files);

    p_teca_binary_segmentation seg = teca_binary_segmentation::New();
    seg->set_threshold_variable(var);
    seg->set_low_threshold_value(low);
    seg->set_high_threshold_value(high);
    seg->set_threshold_by_percentile();
    seg->set_input_connection(cfr->get_output_port());

    // regression test
    if (teca_file_util::file_exists(baseline.c_str()))
    {
        // run the test
        p_teca_cartesian_mesh_reader rea = teca_cartesian_mesh_reader::New();
        rea->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, rea->get_output_port());
        diff->set_input_connection(1, seg->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;
        p_teca_cartesian_mesh_writer wri = teca_cartesian_mesh_writer::New();
        wri->set_input_connection(seg->get_output_port());
        wri->set_file_name(baseline.c_str());
        wri->update();
    }

    return 0;
}
