#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_coordinate_transform.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_index_executive.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_util.h"
#include "teca_system_interface.h"
#include "teca_array_attributes.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <cmath>
#include <functional>

using namespace teca_variant_array_util;


// generates f = k*nxy + j*nx + i
struct index_function
{
    p_teca_variant_array operator()(int, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double t)
    {
        (void)t;

        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();
        size_t nxyz = nx*ny*nz;

        p_teca_variant_array f = x->new_instance(nxyz);

        VARIANT_ARRAY_DISPATCH(f.get(),
            auto [pf] = data<TT>(f);
            for (size_t i = 0; i < nxyz; ++i)
            {
                pf[i] = i;
            }
            )

        return f;
    }
};


int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc < 2)
    {
        std::cerr << "usage: test_cartesian_mesh_coordinate_transform [baseline]" << std::endl;
        return -1;
    }

    std::string baseline = argv[1];

    p_teca_cartesian_mesh_source src = teca_cartesian_mesh_source::New();
    src->set_whole_extents({0, 9, 0, 9, 0, 0, 0, 0});
    src->set_bounds({-1.0, 1.0, -90.0, 90.0, 9500.0, 0.0, 0.0, 0.0});
    src->set_calendar("standard", "days since 2020-04-17 00:00:00");

    index_function func;

    src->append_field_generator({"index",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1}, "unitless",
            "index", "some test data"),
        func});

    p_teca_cartesian_mesh_coordinate_transform coord_tfm =
        teca_cartesian_mesh_coordinate_transform::New();

    coord_tfm->set_input_connection(src->get_output_port());
    coord_tfm->set_target_bounds({0.0, 360.0, 1.0, 0.0, 0.0, 0.0});

    coord_tfm->set_x_axis_variable("longitude");
    coord_tfm->set_y_axis_variable("latitude");
    coord_tfm->set_z_axis_variable("level");

    coord_tfm->set_x_axis_units("degrees_east");
    coord_tfm->set_y_axis_units("degrees_north");
    coord_tfm->set_z_axis_units("hPa");


    // run the test
    p_teca_index_executive exe = teca_index_executive::New();

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(baseline.c_str()))
    {
        std::cerr << "running the test..." << std::endl;

        exe->set_arrays({"index"});

        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->append_file_name(baseline);
        cfr->set_x_axis_variable("longitude");
        cfr->set_y_axis_variable("latitude");

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, cfr->get_output_port());
        diff->set_input_connection(1, coord_tfm->get_output_port());
        diff->set_executive(exe);
        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline..." << std::endl;

        p_teca_cf_writer cfw = teca_cf_writer::New();
        cfw->set_input_connection(coord_tfm->get_output_port());
        cfw->set_file_name(baseline);
        cfw->set_point_arrays({"index"});
        cfw->set_executive(exe);
        cfw->set_thread_pool_size(1);
        cfw->update();
    }


    return 0;
}
