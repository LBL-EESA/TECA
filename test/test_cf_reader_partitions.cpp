#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_util.h"
#include "teca_system_interface.h"
#include "teca_index_executive.h"
#include "teca_space_time_executive.h"
#include "teca_spatial_executive.h"
#include "teca_mpi_manager.h"

#include <vector>
#include <string>
#include <iostream>

int main(int argc, char **argv)
{
    teca_mpi_manager man(argc, argv);
    int rank = man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    if (argc < 7)
    {
        if (rank == 0)
        {
            std::cerr << std::endl << "Usage error:" << std::endl
                << "test_cf_reader_partitions [in files] [partitioner]"
                " [n spatial partitions] [n temproal partitions]"
                " [temporal partition size] [array 0] .. [array n]"
                << std::endl << std::endl;
        }
        return -1;
    }

    std::string in_files = argv[1];
    std::string partitioner = argv[2];
    int n_spatial_partitions = atoi(argv[3]);
    int n_temporal_partitions = atoi(argv[4]);
    int temporal_partition_size = atoi(argv[5]);
    std::vector<std::string> arrays;
    for (int i = 6; i < argc; ++i)
    {
        arrays.push_back(argv[i]);
    }

    p_teca_algorithm_executive exec;
    if (partitioner == "temporal")
    {
        p_teca_index_executive idx_exec = teca_index_executive::New();
        idx_exec->set_arrays(arrays);

        exec = idx_exec;
    }
    else if (partitioner == "space_time")
    {
        p_teca_space_time_executive spt_exec = teca_space_time_executive::New();
        spt_exec->set_number_of_spatial_partitions(n_spatial_partitions);
        spt_exec->set_number_of_temporal_partitions(n_temporal_partitions);
        spt_exec->set_temporal_partition_size(temporal_partition_size);
        spt_exec->set_arrays(arrays);

        exec = spt_exec;
    }
    else if (partitioner == "spatial")
    {
        p_teca_spatial_executive sp_exec = teca_spatial_executive::New();
        //sp_exec->set_number_of_spatial_partitions(n_spatial_partitions);
        sp_exec->set_number_of_temporal_partitions(n_temporal_partitions);
        sp_exec->set_temporal_partition_size(temporal_partition_size);
        sp_exec->set_arrays(arrays);

        exec = sp_exec;
    }
    else
    {
        TECA_ERROR("Invalid partitioner \"" << partitioner << "\"")
        return -1;
    }
    exec->set_verbose(1);

    p_teca_cf_reader cfr = teca_cf_reader::New();
    cfr->set_files_regex(in_files);

    // run the test
    if (rank == 0)
    {
        std::cerr << "running the test ... " << std::endl;
    }

    p_teca_cf_reader baseline_cfr = teca_cf_reader::New();
    baseline_cfr->set_files_regex(in_files);

    p_teca_dataset_diff diff = teca_dataset_diff::New();
    diff->set_input_connection(0, baseline_cfr->get_output_port());
    diff->set_input_connection(1, cfr->get_output_port());
    diff->set_executive(exec);
    diff->set_verbose(1);

    diff->update();

    return 0;
}
