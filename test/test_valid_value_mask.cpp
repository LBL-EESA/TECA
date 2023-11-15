#include "teca_cartesian_mesh_source.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_coordinate_util.h"
#include "teca_index_executive.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_util.h"
#include "teca_system_interface.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"

#include <cmath>
#include <functional>

using namespace teca_variant_array_util;

// generates f = cos(x)*cos(y) where f >= m_threshold and m_fill otherwise
struct cosx_cosy
{
    cosx_cosy() : m_threshold(0.25), m_fill(1.e20) {}

    cosx_cosy(double threshold, double fill) :
        m_threshold(threshold), m_fill(fill) {}

    p_teca_variant_array operator()(int, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double t)
    {
        (void)t;
        (void)z;

        size_t nx = x->size();
        size_t ny = y->size();
        size_t nxy = nx*ny;

        p_teca_variant_array f = x->new_instance(nxy);

        VARIANT_ARRAY_DISPATCH(x.get(),

            assert_type<CTT>(y);

            auto [pf] = data<TT>(f);
            auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);

            sync_host_access_any(x, y);

            for (size_t j = 0; j < ny; ++j)
            {
                for (size_t i = 0; i < nx; ++i)
                {
                    NT cxcy = cos(px[i])*cos(py[j]);
                    pf[j*nx + i] = cxcy < NT(m_threshold) ? NT(m_fill) : cxcy;
                }
            }
            )

        return f;
    }

    double m_threshold;
    double m_fill;
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc < 4)
    {
        std::cerr << "usage: test_valid_value_mask [threshold]"
            " [fill value] [baseline]" << std::endl;
        return -1;
    }

    double threshold = atof(argv[1]);
    double fill_value = atof(argv[2]);
    std::string baseline = argv[3];

    double pi = M_PI;

    // generate input data cos(x)*cos(y)
    p_teca_cartesian_mesh_source s = teca_cartesian_mesh_source::New();
    s->set_whole_extents({0, 99, 0, 99, 0, 0, 0, 0});
    s->set_bounds({-pi, pi, -pi, pi, 0.0, 0.0, 0.0, 0.0});
    s->set_calendar("standard", "days since 2020-12-15 00:00:00");

    cosx_cosy func(threshold, fill_value);

    s->append_field_generator({"func",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,0,1}, "unitless",
            "func", "test data where z = cos(x)*cos(y) where z >= 0.5",
            1, func.m_fill),
            func});

    // generate the mask
    p_teca_valid_value_mask mask = teca_valid_value_mask::New();
    mask->set_input_connection(s->get_output_port());
    mask->set_verbose(1);

    // run the test
    p_teca_index_executive exe = teca_index_executive::New();

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(baseline.c_str()))
    {
        std::cerr << "running the test..." << std::endl;

        exe->set_arrays({"func", "func_valid"});

        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->append_file_name(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, cfr->get_output_port());
        diff->set_input_connection(1, mask->get_output_port());
        diff->set_executive(exe);
        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline..." << std::endl;

        p_teca_cf_writer cfw = teca_cf_writer::New();
        cfw->set_input_connection(mask->get_output_port());
        cfw->set_file_name(baseline);
        cfw->set_point_arrays({"func", "func_valid"});
        cfw->set_executive(exe);
        cfw->set_thread_pool_size(1);
        cfw->update();
    }


    return 0;
}
