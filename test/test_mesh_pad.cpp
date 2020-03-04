#include "teca_mesh_padding.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_system_interface.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>

using namespace std;


// genrates a field with nxl by nyl tiles, each tile has value of 1
struct ones_grid
{
    p_teca_variant_array operator()(const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &,
        double)
    {
        unsigned long nx = x->size();
        unsigned long ny = y->size();

        unsigned long nxy = nx * ny;
        p_teca_double_array grid = teca_double_array::New(nxy);
        double *p_grid = grid->get();

        for (unsigned int i = 0; i < nxy; ++i)
        {
            p_grid[i] = 1;
        }

        return grid;
    }
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    cerr << "argc: " << argc << endl;
    if (argc != 8)
    {
        cerr << "Usage: test_mesh_pad_layer [nx] [ny] [px_low] [px_high] " <<
         "[py_low] [py_high] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long px_low = atoi(argv[3]);
    unsigned long px_high = atoi(argv[4]);
    unsigned long py_low = atoi(argv[5]);
    unsigned long py_high = atoi(argv[6]);
    string out_file = argv[7];

    p_teca_cartesian_mesh_source source = teca_cartesian_mesh_source::New();
    source->set_whole_extents({0l, nx-1l, 0l, ny-1l, 0, 0, 0, 0});
    source->set_bounds({0., 360., -90.0, 90.0, 0., 0., 1., 1.});

    ones_grid grid = {};
    source->append_field_generator({"ones_grid", grid});

    p_teca_mesh_padding mesh_padder = teca_mesh_padding::New();
    mesh_padder->set_input_connection(source->get_output_port());
    mesh_padder->set_px_low(px_low);
    mesh_padder->set_px_high(px_high);
    mesh_padder->set_py_low(py_low);
    mesh_padder->set_py_high(py_high);

    if (teca_file_util::file_exists(out_file.c_str()))
    {
        // run the test
        p_teca_cartesian_mesh_reader reader = teca_cartesian_mesh_reader::New();
        reader->set_file_name(out_file);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, reader->get_output_port());
        diff->set_input_connection(1, mesh_padder->get_output_port());

        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << out_file << endl;

        p_teca_cartesian_mesh_writer writer = teca_cartesian_mesh_writer::New();
        writer->set_input_connection(mesh_padder->get_output_port());
        writer->set_file_name(out_file);

        // run the pipeline
        writer->update();

        return -1;
    }

    return 0;
}
