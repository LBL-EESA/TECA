#include "teca_config.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_table_reader.h"
#include "teca_table_reduce.h"
#include "teca_table_sort.h"
#include "teca_table_writer.h"
#include "teca_test_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>

using namespace std;

// --------------------------------------------------------------------------
int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    // parse command line
    std::string input;
    std::string baseline;
    int have_baseline = 0;
    std::string index;
    long first_step = 0;
    long last_step = -1;
    unsigned int n_threads = 1;

    if (rank == 0)
    {
        if (argc != 7)
        {
            cerr << endl << "Usage error:" << endl
                << "test_table_reader_distribute [input] [output] "
                   "[index column] [first step] [last step] [n threads]"
                << endl << endl;
            return -1;
        }

        // parse command line
        input = argv[1];
        baseline = argv[2];
        if (teca_file_util::file_exists(baseline.c_str()))
            have_baseline = 1;
        index = argv[3];
        first_step = atoi(argv[4]);
        last_step = atoi(argv[5]);
        n_threads = atoi(argv[6]);
    }

    teca_test_util::bcast(input);
    teca_test_util::bcast(baseline);
    teca_test_util::bcast(have_baseline);
    teca_test_util::bcast(index);
    teca_test_util::bcast(first_step);
    teca_test_util::bcast(last_step);
    teca_test_util::bcast(n_threads);

    // create the pipeline objects
    p_teca_table_reader reader = teca_table_reader::New();
    reader->set_file_name(input);
    reader->set_index_column(index);
    reader->set_generate_original_ids(1);

    // map-reduce
    p_teca_table_reduce map_reduce = teca_table_reduce::New();
    map_reduce->set_input_connection(reader->get_output_port());
    map_reduce->set_first_step(first_step);
    map_reduce->set_last_step(last_step);
    map_reduce->set_verbose(1);
    map_reduce->set_thread_pool_size(n_threads);

    // sort results
    p_teca_table_sort sort = teca_table_sort::New();
    sort->set_input_connection(map_reduce->get_output_port());
    sort->set_index_column("original_ids");

    // regression test
    if (have_baseline)
    {
        // run the test
        p_teca_table_reader baseline_reader = teca_table_reader::New();
        baseline_reader->set_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_reader->get_output_port());
        diff->set_input_connection(1, sort->get_output_port());
        diff->update();
    }
    else
    {
        // make a baseline
        if (rank == 0)
            cerr << "generating baseline image " << baseline << endl;
        p_teca_table_writer table_writer = teca_table_writer::New();
        table_writer->set_input_connection(sort->get_output_port());
        table_writer->set_file_name(baseline.c_str());
        table_writer->update();
    }

    return 0;
}
