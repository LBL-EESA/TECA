#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_cartesian_mesh_subset.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_system_util.h"
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

    if (argc < 24)
    {
        if (rank == 0)
        {
            cerr << endl
                << "Usage:" << endl
                << "test_cartesian_mesh_regrid "
                   "[target regex] [target x axis] [target y axis] [target z axis] [target t axis] "
                   "[n target arrays] [target array 1] ... [target array n] "
                   "[source regex] [source x axis] [source y axis] [source z axis] [source t axis] "
                   "[n source arrays] [source array 1] ... [source array n] "
                   "[output file] [first step] [last step] [target bounds]" << endl
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
    string out_file = argv[++arg];
    unsigned long first_step = atol(argv[++arg]);
    unsigned long last_step = atol(argv[++arg]);
    vector<double> target_bounds(6, 0l);
    for (int i = 0; i < 6; ++i)
       target_bounds[i] = atol(argv[++arg]);

    // work around cmake not passing "" through
    target_x_ax = target_x_ax == "." ? "" : target_x_ax;
    target_y_ax = target_y_ax == "." ? "" : target_y_ax;
    target_z_ax = target_z_ax == "." ? "" : target_z_ax;
    target_t_ax = target_t_ax == "." ? "" : target_t_ax;

    source_x_ax = source_x_ax == "." ? "" : source_x_ax;
    source_y_ax = source_y_ax == "." ? "" : source_y_ax;
    source_z_ax = source_z_ax == "." ? "" : source_z_ax;
    source_t_ax = source_t_ax == "." ? "" : source_t_ax;

    // create the target dataset reader
    p_teca_cf_reader tr = teca_cf_reader::New();
    tr->set_x_axis_variable(target_x_ax);
    tr->set_y_axis_variable(target_y_ax);
    tr->set_z_axis_variable(target_z_ax);
    tr->set_t_axis_variable(target_t_ax);
    tr->set_files_regex(target_regex);

    p_teca_normalize_coordinates tc = teca_normalize_coordinates::New();
    tc->set_input_connection(tr->get_output_port());

    // create the source dataset reader
    p_teca_cf_reader sr = teca_cf_reader::New();
    sr->set_x_axis_variable(source_x_ax);
    sr->set_y_axis_variable(source_y_ax);
    sr->set_z_axis_variable(source_z_ax);
    sr->set_t_axis_variable(source_t_ax);
    sr->set_files_regex(source_regex);

    p_teca_normalize_coordinates sc = teca_normalize_coordinates::New();
    sc->set_input_connection(sr->get_output_port());

    // create the regrider
    p_teca_cartesian_mesh_regrid rg = teca_cartesian_mesh_regrid::New();
    rg->set_input_connection(0, tc->get_output_port());
    rg->set_input_connection(1, sc->get_output_port());
    rg->set_arrays(source_arrays);

    // create the subseter
    p_teca_cartesian_mesh_subset ss = teca_cartesian_mesh_subset::New();
    ss->set_input_connection(rg->get_output_port());
    ss->set_bounds(target_bounds);

    // set the executive on the writer to stream time steps
    p_teca_index_executive exec = teca_index_executive::New();

    // optional
    exec->set_start_index(first_step);
    exec->set_end_index(last_step);
    exec->set_arrays(target_arrays);

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(out_file.c_str()))
    {
        // run the test
        p_teca_cartesian_mesh_reader baseline_reader = teca_cartesian_mesh_reader::New();
        baseline_reader->set_file_name(out_file);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_reader->get_output_port());
        diff->set_input_connection(1, ss->get_output_port());
        diff->set_executive(exec);

        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << out_file << endl;

        p_teca_cartesian_mesh_writer baseline_writer = teca_cartesian_mesh_writer::New();
        baseline_writer->set_input_connection(ss->get_output_port());
        baseline_writer->set_file_name(out_file);

        baseline_writer->set_executive(exec);

        // run the pipeline
        baseline_writer->update();

        return -1;
    }

    return 0;
}
