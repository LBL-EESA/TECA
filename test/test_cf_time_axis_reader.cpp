#include "teca_common.h"
#include "teca_algorithm.h"
#include "teca_dataset.h"
#include "teca_netcdf_util.h"
#include "teca_mpi.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_dataset_capture.h"
#include "teca_index_executive.h"
#include "teca_index_reduce.h"
#include "teca_cf_time_axis_reader.h"
#include "teca_cf_time_axis_data_reduce.h"
#include "teca_mpi_util.h"

#include <iostream>

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int n_ranks = mpi_man.get_comm_size();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    std::string regex = argv[1];
    int max_comm_size = atoi(argv[2]);

    // subset the communicator
    MPI_Comm comm = MPI_COMM_WORLD;
    if (n_ranks > max_comm_size)
    {
        teca_mpi_util::equipartition_communicator(
            MPI_COMM_WORLD, max_comm_size, &comm);
    }

    p_teca_cf_time_axis_reader read = teca_cf_time_axis_reader::New();
    read->set_communicator(comm);
    read->set_files_regex(regex);

    p_teca_cf_time_axis_data_reduce reduce =
        teca_cf_time_axis_data_reduce::New();

    reduce->set_communicator(comm);
    reduce->set_input_connection(read->get_output_port());
    reduce->set_verbose(1);
    reduce->set_thread_pool_size(1);

    p_teca_dataset_capture ds_cap = teca_dataset_capture::New();
    ds_cap->set_communicator(comm);
    ds_cap->set_input_connection(reduce->get_output_port());

    ds_cap->update();

    const_p_teca_dataset ds = ds_cap->get_dataset();

    if (ds)
        ds->to_stream(std::cerr);

    return 0;
}
