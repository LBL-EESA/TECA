#include "teca_config.h"
#include "teca_wrf_reader.h"
#include "teca_normalize_coordinates.h"
#include "teca_face_to_cell_centering.h"
#include "teca_vertical_coordinate_transform.h"
#include "teca_cartesian_mesh_writer.h"
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
    const p_teca_wrf_reader &wrf_reader,
    const p_teca_cartesian_mesh_writer &vtk_writer,
    const p_teca_index_executive exec);


int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    // create the pipeline objects
    p_teca_wrf_reader wrf_reader = teca_wrf_reader::New();
    p_teca_face_to_cell_centering face_to_cell = teca_face_to_cell_centering::New();
    p_teca_vertical_coordinate_transform eta_to_p = teca_vertical_coordinate_transform::New();
    p_teca_cartesian_mesh_writer vtk_writer = teca_cartesian_mesh_writer::New();
    p_teca_index_executive exec = teca_index_executive::New();

    // initialize them from command line options
    if (parse_command_line(argc, argv, rank, wrf_reader, vtk_writer, exec))
        return -1;

#ifdef TECA_DEBUG
    teca_metadata md = wrf_reader->update_metadata();
    md.to_stream(std::cerr);
#endif

    // build the pipeline
    face_to_cell->set_input_connection(wrf_reader->get_output_port());
    eta_to_p->set_input_connection(face_to_cell->get_output_port());
    vtk_writer->set_input_connection(eta_to_p->get_output_port());
    vtk_writer->set_executive(exec);
    vtk_writer->set_binary(1);

    // run the pipeline
    vtk_writer->update();

    return 0;
}


// --------------------------------------------------------------------------
int parse_command_line(int argc, char **argv, int rank,
    const p_teca_wrf_reader &wrf_reader,
    const p_teca_cartesian_mesh_writer &vtk_writer,
    const p_teca_index_executive exec)
{
    if (argc < 3)
    {
        if (rank == 0)
        {
            cerr << endl << "Usage error:" << endl
                << "test_wrf_reader [-i input regex] [-o output] [-s first step,last step] "
                << "[-x x axis variable] [-y y axis variable] [-z z axis variable] "
                << "[-t t axis variable] [-b x0,x1,y0,y1,z0,z1] [-e i0,i1,j0,j1,k0,k1] "
                << "[array 0] ... [array n]"
                << endl << endl;
        }
        return -1;
    }

    string regex;
    string output;
    string x_ax = "";
    string y_ax = "";
    string z_ax = "";
    string t_ax = "";
    string t_template = "";
    long first_step = 0;
    long last_step = -1;
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
        else if (!strcmp("-n", argv[i]))
        {
            t_template = argv[++i];
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
    }

    vector<string> arrays;
    for (int i = 2*j+1; i < argc; ++i)
    {
        arrays.push_back(argv[i]);
    }

    // pass the command line options
    /*if (!x_ax.empty())
        wrf_reader->set_x_cell_axis_variable(x_ax);
    if (!y_ax.empty())
        wrf_reader->set_y_cell_axis_variable(y_ax);
    if (!x_ax.empty())
        wrf_reader->set_z_cell_axis_variable(z_ax);
    if (!t_ax.empty())
        wrf_reader->set_t_axis_variable(t_ax);*/
    wrf_reader->set_files_regex(regex);
    wrf_reader->set_filename_time_template(t_template);

    vtk_writer->set_file_name(output);

    exec->set_start_index(first_step);
    exec->set_end_index(last_step);
    exec->set_extent(extent);
    exec->set_bounds(bounds);
    exec->set_arrays(arrays);

    return 0;
}
