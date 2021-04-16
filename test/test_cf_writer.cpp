#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_normalize_coordinates.h"
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
    const p_teca_cf_reader &cf_reader,
    const p_teca_cf_writer &cf_writer,
    const p_teca_index_executive exec);


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // create the pipeline objects
    p_teca_cf_reader cf_reader = teca_cf_reader::New();
    p_teca_cf_writer cf_writer = teca_cf_writer::New();
    p_teca_index_executive exec = teca_index_executive::New();

    // initialize them from command line options
    if (parse_command_line(argc, argv, rank, cf_reader, cf_writer, exec))
        return -1;

    // build the pipeline
    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(cf_reader->get_output_port());

    cf_writer->set_input_connection(coords->get_output_port());
    cf_writer->set_executive(exec);

    // run the pipeline
    cf_writer->update();

    return 0;
}


// --------------------------------------------------------------------------
int parse_command_line(int argc, char **argv, int rank,
    const p_teca_cf_reader &cf_reader,
    const p_teca_cf_writer &cf_writer,
    const p_teca_index_executive exec)
{
    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_cf_writer [-i input regex] [-o output] [-s first step,last step] "
                << "[-x x axis variable] [-y y axis variable] [-z z axis variable] "
                << "[-t t axis variable] [-b x0,x1,y0,y1,z0,z1] [-e i0,i1,j0,j1,k0,k1] "
                << "[-c steps per file] [-n num threads] [ -d compression level] "
                << "[array 0] ... [array n]"
                << endl << endl;
        }
        return -1;
    }

    string regex;
    string output;
    string x_ax = "lon";
    string y_ax = "lat";
    string z_ax = "";
    string t_ax = "";
    int n_threads = 1;
    long first_step = 0;
    long last_step = -1;
    long steps_per_file = 1;
    int deflate_level = -1;
    std::vector<double> bounds;
    std::vector<unsigned long> extent;

    int j = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp("-i", argv[i]))
        {
            regex = argv[++i];
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
        else if (!strcmp("-b", argv[i]))
        {
            bounds.resize(6,0.0);
            double *bds = bounds.data();
            sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf,%lf",
                bds, bds+1, bds+2, bds+3, bds+4, bds+5);
            ++j;
        }
        else if (!strcmp("-e", argv[i]))
        {
            extent.resize(6,0.0);
            unsigned long *ext = extent.data();
            sscanf(argv[++i], "%lu,%lu,%lu,%lu,%lu,%lu",
                ext, ext+1, ext+2, ext+3, ext+4, ext+5);
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
        else if (!strcmp("-d", argv[i]))
        {
            deflate_level = atoi(argv[++i]);
            ++j;
        }
    }

    vector<string> arrays;
    for (int i = 2*j+1; i < argc; ++i)
        arrays.push_back(argv[i]);

    // pass the command line options
    cf_reader->set_x_axis_variable(x_ax);
    cf_reader->set_y_axis_variable(y_ax);
    cf_reader->set_z_axis_variable(z_ax);
    cf_reader->set_t_axis_variable(t_ax);
    cf_reader->set_files_regex(regex);

    cf_writer->set_file_name(output);
    cf_writer->set_thread_pool_size(n_threads);
    cf_writer->set_first_step(first_step);
    cf_writer->set_last_step(last_step);
    cf_writer->set_layout(teca_cf_writer::number_of_steps);
    cf_writer->set_steps_per_file(steps_per_file);
    cf_writer->set_compression_level(deflate_level);
    cf_writer->set_point_arrays(arrays);

    exec->set_extent(extent);
    exec->set_bounds(bounds);
    exec->set_arrays(arrays);
    exec->set_verbose(1);

    return 0;
}
