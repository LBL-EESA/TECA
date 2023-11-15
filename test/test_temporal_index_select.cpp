#include "teca_cf_reader.h"
#include "teca_temporal_index_select.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_spatial_executive.h"
#include "teca_space_time_executive.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
#include "teca_mpi_manager.h"

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

    enum partition_type
    {
        INDEX = 0,
        SPATIAL = 1,
        SPACE_TIME = 2,

    };

    // parse the command line
    if (argc < 5)
    {
        if ( rank == 0 )
        {
            std::cerr << std::endl << "Usage error:" << std::endl
                << "test_temporal_index_select [input regex] [baseline]"
                " [array_name] [partitioning_code] [index]"
                " [index] ..." << std::endl
                << std::endl;
            return -1;
        }
    }
    std::string regex = argv[1];
    std::string baseline = argv[2];
    std::vector<std::string> point_arrays = {argv[3]};
    int partitioning = atoi(argv[4]);
    std::vector<long long> indices;
    indices.push_back(atoi(argv[5]));
    for (auto i = 6; i < argc; ++i)
    {
        indices.push_back(atoi(argv[i]));
    }

    // create the cf reader
    p_teca_cf_reader reader = teca_cf_reader::New();
    reader->set_files_regex(regex);

    // temporal index select
    p_teca_temporal_index_select select = teca_temporal_index_select::New();
    select->set_input_connection(reader->get_output_port());
    select->set_indices(indices);

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test)
    {
        if (rank == 0) std::cerr << "running the test..." << std::endl;

        baseline += ".*\\.nc$";

        p_teca_cf_reader br = teca_cf_reader::New();
        br->set_files_regex(baseline);

        // choose the executive 
        p_teca_algorithm_executive rex;
        p_teca_spatial_executive spatial_exec;
        p_teca_space_time_executive space_time_exec;
        switch (partitioning)
        {
        case SPATIAL:
            spatial_exec = teca_spatial_executive::New();
            spatial_exec->set_arrays(point_arrays);
            spatial_exec->set_number_of_temporal_partitions(0);
            rex = spatial_exec;
            break;
        case SPACE_TIME:
            space_time_exec = teca_space_time_executive::New();
            space_time_exec->set_arrays(point_arrays);
            space_time_exec->set_number_of_temporal_partitions(0);
            rex = space_time_exec;
            break;
        
        default:
            p_teca_index_executive index_exec = teca_index_executive::New();
            index_exec->set_arrays(point_arrays);
            rex = index_exec;
            break;
        }

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, br->get_output_port());
        diff->set_input_connection(1, select->get_output_port());
        diff->set_executive(rex);
        diff->update();
    }
    else
    {
        if (rank == 0) std::cerr << "writing the baseline..." << std::endl;

        baseline += "_%t%.nc";

        // writer
        p_teca_cf_writer writer = teca_cf_writer::New();
        writer->set_input_connection(select->get_output_port());
        writer->set_thread_pool_size(1);
        writer->set_point_arrays(point_arrays);
        writer->set_file_name(baseline);
        writer->set_layout_to_yearly();
        writer->set_partitioner_to_spatial();
        writer->set_number_of_spatial_partitions(0);

        writer->update();
    }

    return 0;
}
