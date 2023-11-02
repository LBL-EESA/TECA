#include "teca_config.h"
#include "teca_l2_norm.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_util.h"
#include "teca_system_interface.h"
#include "teca_index_executive.h"
#include "teca_array_attributes.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <vector>
#include <string>
#include <iostream>

using namespace teca_variant_array_util;

// generates : f(x,y,z) = x - t; or f(x,y,z) = y; or f(x,y,z) = z depending on
// the value of m_dir. this is used to generate a vector field whose magnitude
// is concentric spheres centered at (t,0,0)
struct fxyz
{
    fxyz(char a_dir) : m_dir(a_dir) {}

    char m_dir;

    p_teca_variant_array operator()(int, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double t)
    {
        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();
        size_t nxy = nx*ny;
        size_t nxyz = nxy*nz;

        p_teca_variant_array f = x->new_instance(nxyz);

        VARIANT_ARRAY_DISPATCH_FP(f.get(),

            assert_type<CTT>(y, z);

            auto [pf] = data<TT>(f);
            auto [spx, px, spy, py, spz, pz] = get_host_accessible<CTT>(x, y, z);

            sync_host_access_any(x, y, z);

            for (size_t k = 0; k < nz; ++k)
            {
                for (size_t j = 0; j < ny; ++j)
                {
                    for (size_t i = 0; i < nx; ++i)
                    {
                        NT fval = NT();
                        switch (m_dir)
                        {
                            case 'x':
                                fval = px[i] - NT(t);
                                break;
                            case 'y':
                                fval = py[j];
                                break;
                            case 'z':
                                fval = pz[k];
                                break;
                        }
                        pf[k*nxy + j*nx + i] = fval;
                    }
                }
            }
            )

        return f;
    }
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 2)
    {
        std::cerr << std::endl << "Usage error:" << std::endl
            << "test_l2_norm [out file]" << std::endl << std::endl;
        return -1;
    }

    size_t nx = 32;
    size_t ny = 16;
    size_t nz = 8;
    size_t nt = 1;
    std::string out_file = argv[1];


    p_teca_cartesian_mesh_source mesh = teca_cartesian_mesh_source::New();
    mesh->set_whole_extents({0, nx - 1, 0, ny - 1, 0, nz - 1, 0, nt - 1});
    mesh->set_bounds({-4.0, 4.0, -2.0, 2.0, -1.0, 1.0, -1.0, 1.0});
    mesh->set_calendar("standard", "days since 2021-11-11 00:00:00");

    mesh->append_field_generator({"U",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1}, "meters",
            "distance", "distance from x=0"),
            fxyz('x')});

    mesh->append_field_generator({"V",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1}, "meters",
            "distance", "distance from y = 0"),
            fxyz('y')});

    mesh->append_field_generator({"W",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1}, "meters",
            "distance", "distance from z = 0"),
            fxyz('z')});


    p_teca_l2_norm l2 = teca_l2_norm::New();
    l2->set_input_connection(mesh->get_output_port());
    l2->set_component_0_variable("U");
    l2->set_component_1_variable("V");
    l2->set_component_2_variable("W");
    l2->set_l2_norm_variable("R");

    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_arrays({"U","V","W","R"});

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(out_file.c_str()))
    {
        // run the test
        std::cerr << "running the test ... " << std::endl;

        p_teca_cartesian_mesh_reader baseline_reader = teca_cartesian_mesh_reader::New();
        baseline_reader->set_file_name(out_file);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, baseline_reader->get_output_port());
        diff->set_input_connection(1, l2->get_output_port());
        diff->set_executive(exec);
        diff->set_verbose(1);

        diff->update();
    }
    else
    {
        // make a baseline
        std::cerr << "generating baseline image ... " << out_file << std::endl;

        p_teca_cartesian_mesh_writer baseline_writer = teca_cartesian_mesh_writer::New();
        baseline_writer->set_input_connection(l2->get_output_port());
        baseline_writer->set_file_name(out_file);
        baseline_writer->set_executive(exec);

        // run the pipeline
        baseline_writer->update();

        return -1;
    }

    return 0;
}
