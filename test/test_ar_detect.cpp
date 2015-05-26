#include "teca_cf_reader.h"
#include "teca_temporal_average.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_ar_detect.h"
#include "teca_table_reduce.h"
#include "teca_table_writer.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

int main(int argc, char **argv)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (argc < 9)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_cf_reader [input dataset regex] [water vapor var] "
                   "[mask dataset regex] [mask var] "
                   "[output base] [filter width] [first step] [last step]" << endl
                << endl;
        }
        return -1;
    }

    // parse command line
    string data_regex = argv[1];
    string vapor_var = argv[2];
    string mask_regex = argv[3];
    string mask_var = argv[4];
    string output_base = argv[5];
    int filter_width = atoi(argv[6]);
    long first_step = atol(argv[7]);
    long last_step = atol(argv[8]);

    // dataset reader
    p_teca_cf_reader d_cfr = teca_cf_reader::New();
    d_cfr->set_files_regex(data_regex);

    p_teca_algorithm alg_0 = d_cfr;
    if (filter_width)
    {
        // temporal average input data
        p_teca_temporal_average avg = teca_temporal_average::New();
        avg->set_input_connection(d_cfr->get_output_port());
        avg->set_filter_width(filter_width);
        alg_0 = avg;
    }

    p_teca_algorithm alg_1 = alg_0;
    if (!mask_regex.empty())
    {
        // land sea mask reader
        p_teca_cf_reader m_cfr = teca_cf_reader::New();
        m_cfr->set_t_axis_variable("");
        m_cfr->set_files_regex(mask_regex);

        // regrid land sea mask onto water vaopr data
        p_teca_cartesian_mesh_regrid cmr = teca_cartesian_mesh_regrid::New();
        cmr->set_input_connection(0, alg_0->get_output_port());
        cmr->set_input_connection(1, m_cfr->get_output_port());
        cmr->add_array(mask_var);
        alg_1 = cmr;
    }

    // ar detect
    p_teca_ar_detect ard = teca_ar_detect::New();
    ard->set_input_connection(alg_1->get_output_port());
    ard->set_water_vapor_variable(vapor_var);
    ard->set_land_sea_mask_variable(mask_var);

    // reduction
    p_teca_table_reduce tr = teca_table_reduce::New();
    tr->set_input_connection(ard->get_output_port());
    tr->set_first_step(first_step);
    tr->set_last_step(last_step);
    tr->set_thread_pool_size(4);

    // writer
    p_teca_table_writer tw = teca_table_writer::New();
    tw->set_input_connection(tr->get_output_port());
    tw->set_base_file_name(output_base);

    // run the pipeline
    tw->update();

#if defined(TECA_HAS_MPI)
    MPI_Finalize();
#endif
    return 0;
}
