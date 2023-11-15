#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_elevation_mask.h"
#include "teca_integrated_water_vapor.h"
#include "teca_index_executive.h"
#include "teca_coordinate_util.h"
#include "teca_dataset_capture.h"
#include "teca_system_interface.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"

#include <cmath>
#include <functional>

using namespace teca_variant_array_util;

// iwv = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} q dp
//
// strategy: define integrand q = exp(p) and evaluate numerically using our
// implementation and analytically and compare the results as a check
//
// let q = exp(p)
//
// we can then validate our implementation against the analyitic solutions:
//
// iwv = exp(p) |_{p_{sfc}}^{p_{top}}
//

template <typename num_t, typename array_t = teca_variant_array_impl<num_t>>
struct function_of_z
{
    using f_type = std::function<num_t(num_t)>;

    function_of_z(const f_type &a_f, num_t max_z, num_t fill_value) :
        m_max_z(max_z), m_fill_value(fill_value), m_f(a_f)  {}

    p_teca_variant_array operator()(int, const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double t)
    {
        (void)t;

        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();
        size_t nxy = nx*ny;

        auto [fz, pfz] = ::New<array_t>(nx*ny*nz);

        VARIANT_ARRAY_DISPATCH(z.get(),
            auto [spz, pz] = get_host_accessible<CTT>(z);
            sync_host_access_any(z);
            for (size_t k = 0; k < nz; ++k)
            {
                for (size_t j = 0; j < ny; ++j)
                {
                    for (size_t i = 0; i < nx; ++i)
                    {
                        num_t z = pz[k];
                        pfz[k*nxy + j*nx + i] = z > m_max_z ? m_fill_value : this->m_f(z);
                    }
                }
            }
            )

        return fz;
    }

    num_t m_max_z;
    num_t m_fill_value;
    f_type m_f;
};

// an actual sequence of pressure levels from the HighResMIP data
// plev = 92500, 85000, 70000, 60000, 50000, 25000, 5000 ;

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    teca_system_interface::set_stack_trace_on_error();

    unsigned long i1 = 1024;
    double p_sfc = 92500e-4;
    double p_top = 5000e-4;
    double fill_value = 1.0e14;
    int vv_mask = (argc > 0 ? atoi(argv[1]) : 1);
    int write_input = (argc > 1 ? atoi(argv[2]) : 0);
    int write_output = (argc > 2 ? atoi(argv[3]) : 0);

    // double the z axis, but hit all of the original points.  if we set the
    // integrand to the fill_value where p > p_sfc and apply the valid value
    // mask then the integral should have the value as if integrated from p_sfc
    // to p_top. This lets us verify that the integrator works correctly in
    // the presence of missing values
    double p_sfc_2 = 2.0*p_sfc - p_top;
    unsigned long j1 = 2*i1;

    p_teca_cartesian_mesh_source mesh = teca_cartesian_mesh_source::New();
    mesh->set_whole_extents({0, 2, 0, 2, 0, j1, 0, 0});
    mesh->set_bounds({-1.0, 1.0, -1.0, 1.0, p_sfc_2, p_top, 0.0, 0.0});
    mesh->set_calendar("standard", "days since 2020-09-30 00:00:00");

    // let q = exp(p)
    function_of_z<double> q([](double p) -> double { return exp(p); },
        p_sfc, fill_value);

    mesh->append_field_generator({"q",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, {1,1,1,1},
            "g kg^-1", "specific humidty", "test data where q = sin(p)",
            1, fill_value),
            q});

    p_teca_algorithm head;
    if (vv_mask)
    {
        // generate the valid value mask
        std::cerr << "Testing with the valid_value_mask" << std::endl;

        p_teca_valid_value_mask mask = teca_valid_value_mask::New();
        mask->set_input_connection(mesh->get_output_port());
        mask->set_verbose(0);

        head = mask;
    }
    else
    {
        // generate the elevation mask
        std::cerr << "Testing with the elevation_mask" << std::endl;

        // add generator for mesh height
        // let zg = -p
        function_of_z<double> zg([](double z) -> double { return -z; },
                1e6, fill_value);

        mesh->append_field_generator({"zg",
            teca_array_attributes(teca_variant_array_code<double>::get(),
                teca_array_attributes::point_centering, 0, {1,1,1,1}, "m",
                "mesh height", "test data where zg = p",
                1, fill_value),
                zg});

        // generate surface elevation
        p_teca_cartesian_mesh_source elev = teca_cartesian_mesh_source::New();
        elev->set_whole_extents({0, 2, 0, 2, 0, 0, 0, 0});
        elev->set_bounds({-1.0, 1.0, -1.0, 1.0, p_sfc, p_sfc, 0.0, 0.0});
        elev->set_t_axis_variable("");

        elev->append_field_generator({"z",
            teca_array_attributes(teca_variant_array_code<double>::get(),
                teca_array_attributes::point_centering, 0, {1,1,0,1}, "m",
                "surface elevation", "test data where z = p",
                1, fill_value),
                zg});

        p_teca_elevation_mask mask = teca_elevation_mask::New();
        mask->set_input_connection(0, mesh->get_output_port());
        mask->set_input_connection(1, elev->get_output_port());
        mask->set_mask_variables({"q_valid"});
        mask->set_surface_elevation_variable("z");
        mask->set_mesh_height_variable("zg");

        head = mask;
    }

    // write the test input dataset
    if (write_input)
    {
        std::string fn = std::string("test_integrated_water_vapor_input_") +
            std::string(vv_mask ? "vv_mask" : "elev_mask") + std::string("_%t%.nc");

        p_teca_index_executive exec = teca_index_executive::New();

        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(head->get_output_port());
        w->set_file_name(fn);
        w->set_thread_pool_size(1);
        w->set_point_arrays({"q", "q_valid"});
        if (!vv_mask)
            w->append_point_array("zg");
        w->set_executive(exec);

        w->update();
    }

    // compute IWV
    p_teca_integrated_water_vapor iwv = teca_integrated_water_vapor::New();
    iwv->set_input_connection(head->get_output_port());
    iwv->set_specific_humidity_variable("q");
    iwv->set_iwv_variable("iwv");

    // write the result
    if (write_output)
    {
        std::string fn = std::string("test_integrated_water_vapor_output_") +
            std::string(vv_mask ? "vv_mask" : "elev_mask") + std::string("_%t%.nc");

        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(iwv->get_output_port());
        w->set_file_name(fn);
        w->set_thread_pool_size(1);
        w->set_point_arrays({"iwv"});
        w->update();
    }

    // capture the result
    p_teca_dataset_capture dsc = teca_dataset_capture::New();
    dsc->set_input_connection(iwv->get_output_port());
    dsc->update();

    const_p_teca_cartesian_mesh m =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(dsc->get_dataset());

    const_p_teca_double_array iwva =
        std::dynamic_pointer_cast<const teca_double_array>
            (m->get_point_arrays()->get("iwv"));

    double test_iwv = iwva->get(0);

    // calculate the analytic solution
    double g = 9.80665;
    double base_iwv = -(1./g) * (exp(p_top) - exp(p_sfc));

    // display
    std::cerr << "base_iwv = " << base_iwv << std::endl;
    std::cerr << "test_iwv = " << test_iwv << std::endl << std::endl;

    // compare against the analytic solution
    if (!teca_coordinate_util::equal(base_iwv, test_iwv, 1e-4))
    {
        TECA_ERROR("base_iwv = " << base_iwv << " test_iwv = " << test_iwv)
        return -1;
    }

    return 0;
}
