#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_mesh_join.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_mpi.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int parse_command_line(
    int argc,
    char **argv,
    int rank,
    const p_teca_cf_reader &cf_reader_1,
    const p_teca_cf_reader &cf_reader_2,
    const p_teca_cf_reader &cf_reader_3,
    const p_teca_cf_writer &cf_writer,
    const p_teca_index_executive exec);


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // create the pipeline objects
    p_teca_cf_reader cf_reader_1 = teca_cf_reader::New();
    p_teca_cf_reader cf_reader_2 = teca_cf_reader::New();
    p_teca_cf_reader cf_reader_3 = teca_cf_reader::New();
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    p_teca_index_executive exec = teca_index_executive::New();

    // initialize them from command line options
    if (parse_command_line(argc, argv, rank, cf_reader_1,
                   cf_reader_2, cf_reader_3, cf_writer, exec))
        return -1;

    p_teca_mesh_join join = teca_mesh_join::New();
    join->set_number_of_input_connections(3);
    join->set_verbose(1);
    join->set_input_connection(0, cf_reader_1->get_output_port());
    join->set_input_connection(1, cf_reader_2->get_output_port());
    join->set_input_connection(2, cf_reader_3->get_output_port());

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(join->get_output_port());

    cf_writer->set_input_connection(coords->get_output_port());
    cf_writer->set_executive(exec);

    // run the pipeline
    cf_writer->update();

    return 0;
}


// --------------------------------------------------------------------------
int parse_command_line(int argc, char **argv, int rank,
    const p_teca_cf_reader &cf_reader_1,
    const p_teca_cf_reader &cf_reader_2,
    const p_teca_cf_reader &cf_reader_3,
    const p_teca_cf_writer &cf_writer,
    const p_teca_index_executive exec)
{
    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_cf_writer [-i input1 input2 input3] [-o output] [-s first step,last step] "
                << "[-x x axis variable] [-y y axis variable] [-z z axis variable] "
                << "[-t t axis variable] [-c steps per file] [-n num threads] "
                << "[-p var0 var1 ...]"
                << endl << endl;
        }
        return -1;
    }

    int n_files = 3;
    vector<string> input;
    string output;
    string x_ax = "lon";
    string y_ax = "lat";
    string z_ax = "";
    string t_ax = "";
    int n_threads = -1;
    long first_step = 0;
    long last_step = -1;
    long steps_per_file = 1;
    vector<string> arrays;

    int j = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp("-i", argv[i]))
        {
            for (int k = 0; k < n_files; ++k)
               input.push_back(argv[++i]);
            ++j;
        }
        else if (!strcmp("-o", argv[i]))
        {
            output = argv[++i];
            ++j;
        }
        else if (!strcmp("-x", argv[i]))
        {
            x_ax = argv[++i];
            ++j;
        }
        else if (!strcmp("-y", argv[i]))
        {
            y_ax = argv[++i];
            ++j;
        }
        else if (!strcmp("-z", argv[i]))
        {
            z_ax = argv[++i];
            ++j;
        }
        else if (!strcmp("-t", argv[i]))
        {
            t_ax = argv[++i];
            ++j;
        }
        else if (!strcmp("-s", argv[i]))
        {
            sscanf(argv[++i], "%li,%li",
                 &first_step, &last_step);
            ++j;
        }
        else if (!strcmp("-n", argv[i]))
        {
            n_threads = atoi(argv[++i]);
            ++j;
        }
        else if (!strcmp("-c", argv[i]))
        {
            steps_per_file = atoi(argv[++i]);
            ++j;
        }
        if (!strcmp("-p", argv[i]))
        {
            for (int k = i; k < argc-1; ++k)
               arrays.push_back(argv[++i]);
            ++j;
        }
    }

    // pass the command line options
    cf_reader_1->set_x_axis_variable(x_ax);
    cf_reader_1->set_y_axis_variable(y_ax);
    cf_reader_1->set_z_axis_variable(z_ax);
    cf_reader_1->set_t_axis_variable(t_ax);
    cf_reader_1->set_files_regex(input[0]);

    cf_reader_2->set_x_axis_variable(x_ax);
    cf_reader_2->set_y_axis_variable(y_ax);
    cf_reader_2->set_z_axis_variable(z_ax);
    cf_reader_2->set_t_axis_variable(t_ax);
    cf_reader_2->set_files_regex(input[1]);

    cf_reader_3->set_x_axis_variable(x_ax);
    cf_reader_3->set_y_axis_variable(y_ax);
    cf_reader_3->set_z_axis_variable(z_ax);
    cf_reader_3->set_t_axis_variable(t_ax);
    cf_reader_3->set_files_regex(input[2]);

    cf_writer->set_file_name(output);
    cf_writer->set_thread_pool_size(n_threads);
    cf_writer->set_first_step(first_step);
    cf_writer->set_last_step(last_step);
    cf_writer->set_layout(teca_cf_writer::number_of_steps);
    cf_writer->set_steps_per_file(steps_per_file);
    cf_writer->set_point_arrays(arrays);

    exec->set_arrays(arrays);

    return 0;
}
