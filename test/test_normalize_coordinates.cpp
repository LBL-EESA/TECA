#include "teca_cartesian_mesh_source.h"
#include "teca_normalize_coordinates.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_dataset_diff.h"
#include "teca_system_interface.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_file_util.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

// genrates a distance field centered on x0,y0,z0
struct distance_field
{
    double m_x0;
    double m_y0;
    double m_z0;

    p_teca_variant_array operator()(const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double)
    {
        unsigned long nx = x->size();
        unsigned long ny = y->size();
        unsigned long nz = z->size();
        unsigned long nxy = nx*ny;

        p_teca_double_array d = teca_double_array::New(nxy*nz);
        double *pd = d->get();

        TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            x.get(),

            const NT *px = std::static_pointer_cast<TT>(x)->get();
            const NT *py = std::static_pointer_cast<TT>(y)->get();
            const NT *pz = std::static_pointer_cast<TT>(z)->get();

            for (unsigned long k = 0; k < nz; ++k)
            {
                NT dz2 = pz[k] - m_z0;
                dz2 *= dz2;
                for (unsigned long j = 0; j < ny; ++j)
                {
                    NT dy2 = py[j] - m_y0;
                    dy2 *= dy2;
                    for (unsigned long i = 0; i < nx; ++i)
                    {
                        NT dx2 = px[i] - m_x0;
                        dx2 *= dx2;
                        pd[k*nxy + j*nx + i] = std::sqrt(dx2 + dy2 + dz2);
                    }
                }
            }
            )

        return d;
    }
};




int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 14)
    {
        cerr << "test_normalize_coordinates [nx] [ny] [nz] [flip x] [flip y] [flip z] "
            "[x0 x1 y0 y1 z0 z1] [out file]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long nz = atoi(argv[3]);
    int flip_x = atoi(argv[4]);
    int flip_y = atoi(argv[5]);
    int flip_z = atoi(argv[6]);
    std::vector<double> req_bounds({atof(argv[7]), atof(argv[8]),
        atof(argv[9]), atof(argv[10]), atof(argv[11]), atof(argv[12])});
    std::string out_file = argv[13];


    p_teca_cartesian_mesh_source source = teca_cartesian_mesh_source::New();
    source->set_whole_extents({0, nx-1, 0, ny-1, 0, nz-1, 0, 0});

    double x0 = flip_x ? 360.0 : 0.0;
    double x1 = flip_x ? 0.0 : 360.0;
    double y0 = flip_y ? 90.0 : -90.0;
    double y1 = flip_y ? -90.0 : 90.0;
    double z0 = flip_z ? 10.0 : 0.0;
    double z1 = flip_z ? 0.0 : 10.0;
    source->set_bounds({x0, x1, y0, y1, z0, z1, 0., 0.});

    distance_field distance = {80., -80., 2.5};
    source->append_field_generator({"distance", distance});

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(source->get_output_port());

    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_bounds(req_bounds);

    if (teca_file_util::file_exists(out_file.c_str()))
    {
        // run the test
        p_teca_cartesian_mesh_reader baseline_reader = teca_cartesian_mesh_reader::New();
        baseline_reader->set_file_name(out_file);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_reader->get_output_port());
        diff->set_input_connection(1, coords->get_output_port());
        diff->set_executive(exec);

        diff->update();
    }
    else
    {
        // make a baseline
        cerr << "generating baseline image " << out_file << endl;

        p_teca_cartesian_mesh_writer baseline_writer = teca_cartesian_mesh_writer::New();
        baseline_writer->set_input_connection(coords->get_output_port());
        baseline_writer->set_file_name(out_file);
        baseline_writer->set_binary(0);
        baseline_writer->set_executive(exec);

        // run the pipeline
        baseline_writer->update();

        return -1;
    }
    return 0;
}
