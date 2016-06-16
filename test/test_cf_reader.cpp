#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_vtk_cartesian_mesh_writer.h"
#include "teca_time_step_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int parse_command_line(
    int argc,
    char **argv,
    int rank,
    const p_teca_cf_reader &cf_reader,
    const p_teca_vtk_cartesian_mesh_writer &vtk_writer,
    const p_teca_time_step_executive exec);


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();

    // create the pipeline objects
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    p_teca_vtk_cartesian_mesh_writer vtk_writer = teca_vtk_cartesian_mesh_writer::New();
    p_teca_time_step_executive exec = teca_time_step_executive::New();

    // initialize them from command line options
    if (parse_command_line(argc, argv, rank, cf_reader, vtk_writer, exec))
        return -1;

    // build the pipeline
    vtk_writer->set_input_connection(cf_reader->get_output_port());
    vtk_writer->set_executive(exec);

    // run the pipeline
    vtk_writer->update();

    return 0;
}


// --------------------------------------------------------------------------
int parse_command_line(
    int argc,
    char **argv,
    int rank,
    const p_teca_cf_reader &cf_reader,
    const p_teca_vtk_cartesian_mesh_writer &vtk_writer,
    const p_teca_time_step_executive exec)
{
    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_cf_reader [input regex] [output] [first step = 0] "
                << "[last step = -1] [x axis = lon] [y axis = lat] [z axis =] [t axis = time] "
                << "[array 0 =] ... [array n =]"
                << endl << endl;
        }
        return -1;
    }

    // parse command line
    string regex = argv[1];
    string output = argv[2];
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
        z_ax = argv[7][0] == '.' ? "" : argv[7];
    string t_ax = "";
    if (argc > 8)
        t_ax = argv[8][0] == '.' ? "" : argv[8];
    vector<string> arrays;
    for (int i = 9; i < argc; ++i)
        arrays.push_back(argv[i]);

    // pass the command line options
    cf_reader->set_x_axis_variable(x_ax);
    cf_reader->set_y_axis_variable(y_ax);
    cf_reader->set_z_axis_variable(z_ax);
    cf_reader->set_t_axis_variable(t_ax);
    cf_reader->set_files_regex(regex);

    vtk_writer->set_file_name(output);

    exec->set_first_step(first_step);
    exec->set_last_step(last_step);
    exec->set_arrays(arrays);

    return 0;
}
