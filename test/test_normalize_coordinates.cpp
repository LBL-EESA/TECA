#include "teca_cartesian_mesh_source.h"
#include "teca_array_attributes.h"
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
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_file_util.h"
#include "teca_system_util.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace teca_variant_array_util;

// genrates a distance field centered on x0,y0,z0
struct distance_field
{
    double m_x0;
    double m_y0;
    double m_z0;

    teca_array_attributes get_attributes()
    {
        std::ostringstream oss;
        oss << "distance to (" << m_x0 << ", " << m_y0 << ", " << m_z0 << ")";

        return teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1}, "degrees",
            "distance", oss.str().c_str());
    }

    p_teca_variant_array operator()(int, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double)
    {
        unsigned long nx = x->size();
        unsigned long ny = y->size();
        unsigned long nz = z->size();
        unsigned long nxy = nx*ny;

        auto [d, pd] = ::New<teca_double_array>(nxy*nz);

        VARIANT_ARRAY_DISPATCH_FP(x.get(),

            assert_type<TT>(y,z);
            auto [spx, px, spy, py, spz, pz] = get_host_accessible<CTT>(x, y, z);

            sync_host_access_any(x, y, z);

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

    if ((argc != 17) && (argc != 11))
    {
        std::cerr << "test_normalize_coordinates [nx] [ny] [nz]"
            " [dataset bounds : x0 x1 y0 y1 z0 z1] [subset bounds : x0 x1 y0 y1 z0 z1]"
            " [out file]" << std::endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long nz = atoi(argv[3]);

    std::vector<double> bounds({atof(argv[4]), atof(argv[5]),
        atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9])});

    // optional subset
    std::vector<double> req_bounds(6);
    bool subset = false;
    if (argc == 17)
    {
        req_bounds = std::vector<double>({atof(argv[10]), atof(argv[11]),
            atof(argv[12]), atof(argv[13]), atof(argv[14]), atof(argv[15])});

        subset = true;
    }

    std::string out_file = argv[subset ? 16 : 10];

    p_teca_cartesian_mesh_source source = teca_cartesian_mesh_source::New();
    source->set_whole_extents({0, nx-1, 0, ny-1, 0, nz-1, 0, 0});

    source->set_bounds({bounds[0], bounds[1], bounds[2],
        bounds[3], bounds[4], bounds[5], 0., 0.});

    distance_field distance = {80., -80., 2.5};
    source->append_field_generator({"distance", distance.get_attributes(), distance});

    p_teca_normalize_coordinates coords = teca_normalize_coordinates::New();
    coords->set_input_connection(source->get_output_port());
    coords->set_enable_periodic_shift_x(1);

    p_teca_index_executive exec = teca_index_executive::New();
    if (subset)
        exec->set_bounds(req_bounds);
    exec->set_verbose(1);

    std::cerr << "running the test with " << std::endl
        << "whole_extents = [0, " << nx-1 << ", 0, "
        << ny-1 << ", 0, " << nz-1 << "]" << std::endl
        << "bounds = [" << bounds[0] << ", " << bounds[1] << ", " << bounds[2]
        << ", " << bounds[3] << ", " << bounds[4] << ", " << bounds[5] << "]"
        << std::endl
        << "req_bounds = [" << req_bounds[0] << ", " << req_bounds[1] << ", "
        << req_bounds[2] << ", " << req_bounds[3] << ", " << req_bounds[4]
        << ", " << req_bounds[5] << "]"
        << std::endl;

    teca_metadata md = coords->update_metadata();

    teca_metadata coord_axes;
    md.get("coordinates", coord_axes);

    std::cerr << "coordinates" << std::endl;
    coord_axes.to_stream(std::cerr);
    std::cerr << std::endl;

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(out_file.c_str()))
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
