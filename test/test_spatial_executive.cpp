#include "teca_common.h"
#include "teca_config.h"
#include "teca_spatial_executive.h"
#include "teca_mpi_manager.h"

//#define VISUALIZE_PARTITIONS
#if defined(VISUALIZE_PARTITIONS)
#include "teca_vtk_util.h"
#include <stdio.h>
#endif

#include <iostream>
#include <array>
#include <deque>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    teca_mpi_manager man(argc, argv);
    int rank = man.get_comm_rank();

    if (argc < 5)
    {
        std::cerr << "error: wrong number of arguments" << std::endl
            << "usage: test_spatial_exzecutive [nx] [ny] [nz] [nt] "
               "[temporal mode = \"size\"/\"number\"] "
               "[mode value = n_temporal_partitions/temporal_partition_size]"
            << std::endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long nz = atoi(argv[3]);

    unsigned long number_of_steps = atoi(argv[4]);

    unsigned long temporal_partition_size = 0;
    unsigned long number_of_temporal_partitions = 1;

    if (argc == 7)
    {
        std::string mode = argv[5];
        unsigned long mode_arg = atoi(argv[6]);
        if (mode == "size")
        {
            temporal_partition_size = mode_arg;
            number_of_temporal_partitions = 0;
        }
        else if (mode == "number")
        {
            temporal_partition_size = 0;
            number_of_temporal_partitions = mode_arg;
        }
        else
        {
            std::cerr << "ERROR: invalid mode \"" << mode << "\"" << std::endl;
            return -1;
        }
    }

#if defined(VISUALIZE_PARTITIONS)
    // for visualizations
    char file_name[128];
    sprintf(file_name, "partitions_rank_%d.vtk", rank);

    teca_vtk_util::partition_writer writer(file_name,
        0.0, -90.0, 0.0, 360.0 / (nx - 1), 180.0 / (ny - 1), 1.0);
#endif

    // the total number of mesh cells processed
    // the test is based on recovering this number after the partitioning
    unsigned long n_cells_expected = nx*ny*nz*number_of_steps;

    // set up the metadata object following necessary conventions
    // for use with the executive.
    teca_metadata md;

    std::array<unsigned long,6> whole_extent({0, nx-1, 0, ny-1, 0, nz-1});

    md.set("index_initializer_key", std::string("n_steps"));
    md.set("n_steps", number_of_steps);
    md.set("index_request_key", std::string("step"));
    md.set("whole_extent", whole_extent);

    // create the executive and have it partition the data
    p_teca_spatial_executive exec = teca_spatial_executive::New();

    exec->set_number_of_temporal_partitions(number_of_temporal_partitions);
    exec->set_temporal_partition_size(temporal_partition_size);

    exec->set_verbose(0);

    exec->initialize(MPI_COMM_WORLD, md);

    // obtain the local requests, and extract the number of cells
    // that would be processed as a result of each
    unsigned long n_cells = 0;

    teca_metadata req;
    while ((req = exec->get_next_request()))
    {
        // get the requested subset
        unsigned long step_extent[2] = {0ul};
        req.get("step", step_extent);

        unsigned long extent[6] = {0l};
        req.get("extent", extent);

        // record the number of cells processed
        n_cells += (extent[1] - extent[0] + 1) *
            (extent[3] - extent[2] + 1) * (extent[5] - extent[4] + 1) *
            (step_extent[1] - step_extent[0] + 1);

        // dump the requested extent for inspection
        std::ostringstream oss;
        oss << rank << " step_extent = [" << step_extent << "]"
            << " extent = [" << extent << "]";
        std::cerr << oss.str() << std::endl;

#if defined(VISUALIZE_PARTITIONS)
        writer.add_partition(extent, step_extent, rank);
#endif
}

#if defined(VISUALIZE_PARTITIONS)
    writer.write();
#endif

    // check that we recover the total number of cells after partitioning
#if defined(TECA_HAS_MPI)
    unsigned long tmp = 0;
    MPI_Reduce(&n_cells, &tmp, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    n_cells = tmp;
#endif

    if (rank == 0)
    {
        if (n_cells != n_cells_expected)
        {
            TECA_ERROR("The number of cells in all partitions " << n_cells
                << " does not match the number of cells in the input domain "
                << n_cells_expected)
            return -1;
        }
        else
        {
            std::cerr << "The number of cells in all partitions "
                << "matches the expected value " << n_cells << std::endl;
        }
    }

    return 0;
}
