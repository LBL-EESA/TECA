#include "teca_config.h"
#include "array_source.h"
#include "array_temporal_stats.h"
#include "array_writer.h"
#include "array_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();
    int n_ranks = mpi_man.get_comm_size();

    teca_system_interface::set_stack_trace_on_error();

    if ((rank == 0) && (argc != 1) && (argc != 4))
    {
        TECA_ERROR(
            << "invalid command line arguments. arguments are:" << endl
            << "arg 1 -> n timesteps" << endl
            << "arg 2 -> array size" << endl
            << "arg 3 -> n threads" << endl)
        exit(-1);
    }

    int n_timesteps = 16;
    int array_size = 5;
    int n_threads = 2;

    if (argc == 4)
    {
        n_timesteps = atoi(argv[1]);
        array_size = atoi(argv[2]);
        n_threads = atoi(argv[3]);
    }

    n_timesteps = max(n_timesteps, 1);
    n_threads = max(n_threads, 1);
    array_size = max(array_size, 1);

    // create a pipeline
    if (rank == 0)
    {
        cerr << "creating the pipeline on " << n_ranks << " rank ..." << endl
        << endl
        << "  5 arrays" << endl
        << "  " << left << setw(3) << n_timesteps << " timesteps      " << right << setw(2) << n_threads << " threads" << endl
        << "  array size " << left << setw(4) << array_size << "      array_1" << endl
        << "      |                  |" << endl
        << "array_source --> array_temporal_stats --> array_writer" << endl
        << endl;
    }

    p_array_source src = array_source::New();
    src->set_number_of_timesteps(n_timesteps);
    src->set_number_of_arrays(5);
    src->set_array_size(array_size);

    p_array_temporal_stats stats = array_temporal_stats::New();
    stats->set_verbose(1);
    stats->set_thread_pool_size(n_threads);
    stats->set_array_name("array_1");

    p_array_writer wri = array_writer::New();

    stats->set_input_connection(src->get_output_port());
    wri->set_input_connection(stats->get_output_port());

    // execute
    cerr << "execute..." << endl;
    wri->update();
    cerr << endl;

    // TODO comapre output and return pass fail code

    return 0;
}
