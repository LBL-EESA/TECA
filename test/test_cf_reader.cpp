#include "teca_cf_reader.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

// example use
// ./test/test_cf_reader ~/work/teca/data/'cam5_1_amip_run2.cam2.h2.1991-10-.*' tmp 0 -1 PS

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << endl << "Usage error:" << endl
            << "test_cf_reader [input regex] [output base] [first step = 0] "
            << "[last step = -1] [x axis = lon] [y axis = lat] [z axis =] [t axis = time] "
            << "[array 0 =] ... [array n =]"
            << endl << endl;
        return -1;
    }

    // parse command line
    string regex = argv[1];
    string base = argv[2];
    long first_step = 0;
    if (argc > 3)
        first_step = atoi(argv[3]);
    long last_step = -1;
    if (argc > 4)
        last_step = atoi(argv[4]);
    string x_ax = "lon";
    if (argc > 5)
        x_ax = argv[5];
    string y_ax = "lat";
    if (argc > 6)
        y_ax = argv[6];
    string z_ax = "";
    if (argc > 7)
        z_ax = argv[7];
    string t_ax = "";
    if (argc > 8)
        t_ax = argv[8];
    vector<string> arrays;
    for (int i = 9; i < argc; ++i)
        arrays.push_back(argv[i]);

    // create the cf reader
    p_teca_cf_reader r = teca_cf_reader::New();
    r->set_x_axis_variable(x_ax);
    r->set_y_axis_variable(y_ax);
    r->set_z_axis_variable(z_ax);
    r->set_t_axis_variable(t_ax);
    r->set_files_regex(regex);

    // create the vtk writer connected to the cf reader
    p_teca_vtk_cartesian_mesh_writer w = teca_vtk_cartesian_mesh_writer::New();
    w->set_base_file_name(base);
    w->set_input_connection(r->get_output_port());

    // set the executive on the writer to stream time steps
    p_teca_time_step_executive exec = teca_time_step_executive::New();
    // optional
    exec->set_first_step(first_step);
    exec->set_last_step(last_step);
    exec->set_arrays(arrays);

    w->set_executive(exec);

    // run the pipeline
    w->update();

    return 0;
}
