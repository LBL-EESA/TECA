#include "teca_cf_reader.h"
#include "teca_cartesian_mesh_subset.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_vtk_cartesian_mesh_writer.h"
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
    if (argc < 24)
    {
        if (rank == 0)
        {
            cerr << endl
                << "Usage:" << endl
                << "test_cf_reader "
                   "[target regex] [target x axis] [target y axis] [target z axis] [target t axis] "
                   "[n target arrays] [target array 1] ... [target array n] "
                   "[source regex] [source x axis] [source y axis] [source z axis] [source t axis] "
                   "[n source arrays] [source array 1] ... [source array n] "
                   "[output base] [first step] [last step] [target bounds]" << endl
                 << endl;
        }
        return -1;
    }

    // parse command line
    int arg = 0;
    string target_regex = argv[++arg];
    string target_x_ax = argv[++arg];
    string target_y_ax = argv[++arg];
    string target_z_ax = argv[++arg];
    string target_t_ax = argv[++arg];
    int n_target_arrays = atoi(argv[++arg]);
    vector<string> target_arrays(argv+arg+1, argv+arg+1+n_target_arrays);
    arg += n_target_arrays;
    string source_regex = argv[++arg];
    string source_x_ax = argv[++arg];
    string source_y_ax = argv[++arg];
    string source_z_ax = argv[++arg];
    string source_t_ax = argv[++arg];
    int n_source_arrays = atoi(argv[++arg]);
    vector<string> source_arrays(argv+arg+1, argv+arg+1+n_source_arrays);
    arg += n_source_arrays;
    string out_base = argv[++arg];
    unsigned long first_step = atol(argv[++arg]);
    unsigned long last_step = atol(argv[++arg]);
    vector<double> target_bounds(6, 0l);
    for (int i = 0; i < 6; ++i)
       target_bounds[i] = atol(argv[++arg]);

    // create the target dataset reader
    p_teca_cf_reader tr = teca_cf_reader::New();
    tr->set_x_axis_variable(target_x_ax);
    tr->set_y_axis_variable(target_y_ax);
    tr->set_z_axis_variable(target_z_ax);
    tr->set_t_axis_variable(target_t_ax);
    tr->set_files_regex(target_regex);

    // create the source dataset reader
    p_teca_cf_reader sr = teca_cf_reader::New();
    sr->set_x_axis_variable(source_x_ax);
    sr->set_y_axis_variable(source_y_ax);
    sr->set_z_axis_variable(source_z_ax);
    sr->set_t_axis_variable(source_t_ax);
    sr->set_files_regex(source_regex);

    // create the regrider
    p_teca_cartesian_mesh_regrid rg = teca_cartesian_mesh_regrid::New();
    rg->set_input_connection(0, tr->get_output_port());
    rg->set_input_connection(1, sr->get_output_port());
    rg->set_source_arrays(source_arrays);

    // create the subseter
    p_teca_cartesian_mesh_subset ss = teca_cartesian_mesh_subset::New();
    ss->set_input_connection(rg->get_output_port());
    ss->set_bounds(target_bounds);

    // create the vtk writer connected to the cf reader
    p_teca_vtk_cartesian_mesh_writer w = teca_vtk_cartesian_mesh_writer::New();
    w->set_base_file_name(out_base);
    w->set_input_connection(ss->get_output_port());

    // set the executive on the writer to stream time steps
    p_teca_time_step_executive exec = teca_time_step_executive::New();
    // optional
    exec->set_first_step(first_step);
    exec->set_last_step(last_step);
    exec->set_arrays(target_arrays);

    w->set_executive(exec);

    // run the pipeline
    w->update();

#if defined(TECA_HAS_MPI)
    MPI_Finalize();
#endif
    return 0;
}
