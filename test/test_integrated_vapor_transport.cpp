#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_elevation_mask.h"
#include "teca_integrated_vapor_transport.h"
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


// ivt = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} q \vec{v} dp
//
// strategy: define integrands that are the product of 2 functions, subsitute in
// q, u, and v and evaluate numerically using our implementation and analytically
// and compare the results as a check
//
// let q = sin a p , u = cos a p, and v = a p.
//
// we can then validate our implementation against the analyitic solutions:
//
// ivt_u = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} cos a p sin a p dp
//       = - \frac{1}{g} [ \frac{1}{2 a} sin^2 a p + c ]
//
// ivt_v = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} a p sin a p dp
//       = - \frac{1}{a^2 g} [ - a^2 p cos a p + sin a p + c ]
//

template <typename num_t>
struct function_of_z
{
    using f_type = std::function<num_t(num_t)>;

    function_of_z(const f_type &a_f, num_t max_z, num_t fill_value) :
        m_max_z(max_z), m_fill_value(fill_value), m_f(a_f)  {}

    p_teca_variant_array operator()(int,
        const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
        const const_p_teca_variant_array &z, double t)
    {
        (void)t;

        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();
        size_t nxy = nx*ny;

        auto [fz, pfz] = New<teca_variant_array_impl<num_t>>(nx*ny*nz);

        VARIANT_ARRAY_DISPATCH(fz.get(),
            assert_type<CTT>(z);
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

    if (argc < 7)
    {
        std::cerr << "test_integrated_vapor_transport [nx] [nz]"
            " [mask : 0 - none, 1 valid value, 2 elevation]"
            " [write_input] [write_output] [trapezoid integrator]"
            << std::endl;
        return -1;
    }

    unsigned int nx = atoi(argv[1]);
    unsigned int nz = atoi(argv[2]);
    int mask = atoi(argv[3]); // 0 - no mask, 1 - vv mask, 2 - elev mask
    int write_input = atoi(argv[4]);
    int write_output = atoi(argv[5]);
    int trapezoid_int = atoi(argv[6]);


    double fill_value = 1.0e14;
    double g = 9.80665;
    double a = 1e-5;
    double p_sfc = 92500.;
    double p_top = 5000.;

    // adjust the lowest and highest pressure level. if we don't do this then
    // the first order integrator error is dominated by the thinkness of the
    // top and bottom layer and is independent of number of points.
    if (!trapezoid_int)
    {
        double dp = 101325.0 / nz;
        p_sfc = 101325.0 + dp;
        p_top = dp;
    }

    // extend the z-axis below the surface to p_bel. Construct the extension to
    // explicitly include the point p_sfc. Set the integrand to the fill_value
    // where p > p_sfc, masking out the points below the surface.  In that case
    // the result is as if we integrated from p_sfc to p_top. This lets us
    // verify that the integrator works correctly in the presence of missing
    // values
    // NOTE: This doesn't currently work with the first order integrator
    // because the integration limits are off. It doesn't hit 101325 exactly.
    double p_bel = p_top + 2. * (p_sfc - p_top) * (double(nz) - 1.) / double(nz);

    p_teca_cartesian_mesh_source mesh = teca_cartesian_mesh_source::New();
    mesh->set_whole_extents({0, nx - 1, 0, nx - 1, 0, nz - 1, 0, 0});

    if (mask)
        mesh->set_bounds({-1.0, 1.0, -1.0, 1.0, p_bel, p_top, 0.0, 0.0});
    else
        mesh->set_bounds({-1.0, 1.0, -1.0, 1.0, p_sfc, p_top, 0.0, 0.0});

    mesh->set_calendar("standard", "days since 2020-09-30 00:00:00");

    std::cerr << "nx = " << nx << " nz = " << nz << std::endl
        << "p_sfc = " << p_sfc << " p_top = " << p_top << " p_bel = " << p_bel  << std::endl
        << "mask type = " << mask << std::endl
        << "write_input = " << write_input << " write_output = " << write_output << std::endl
        << "trapezoid integrator = " << trapezoid_int << std::endl;

    // let q = sin(a p)
    function_of_z<double> q([&a](double p) -> double { return sin(a*p); },
        p_sfc, fill_value);

    mesh->append_field_generator({"q",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyzt_active(), "kg kg-1", "specific humidty",
            "test data where q = sin(a p)", 1, fill_value),
            q});

    // let u = cos(a p)
    function_of_z<double> u([&a](double p) -> double { return cos(a*p); },
        p_sfc, fill_value);

    mesh->append_field_generator({"u",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyzt_active(), "m s^-1", "longitudinal wind velocity",
            "test data where u = cos(a p)", 1, fill_value),
            u});

    // let v = a p
    function_of_z<double> v([&a](double p) -> double { return a*p; },
        p_sfc, fill_value);

    mesh->append_field_generator({"v",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyzt_active(), "m s^-1", "latitudinal wind velocity",
            "test data where v = a p", 1, fill_value),
            v});

    p_teca_algorithm head = mesh;
    if (mask == 1)
    {
        // generate the valid value mask
        std::cerr << "Testing with the valid_value_mask" << std::endl;

        p_teca_valid_value_mask mask = teca_valid_value_mask::New();
        mask->set_input_connection(mesh->get_output_port());
        mask->set_verbose(0);

        head = mask;
    }
    else if (mask == 2)
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
        mask->set_mask_variables({"q_valid", "u_valid", "v_valid"});
        mask->set_surface_elevation_variable("z");
        mask->set_mesh_height_variable("zg");

        head = mask;
    }

    // we need to use the index executive here to enable GPU processing
    p_teca_index_executive exec = teca_index_executive::New();

    // write the test input dataset
    if (write_input)
    {
        std::string fn("test_integrated_vapor_transport_input_");
        if (mask == 1)
            fn += "vv_mask";
        else if (mask == 2)
            fn += "elev_mask";
        else
            fn += "none";
        fn += "_%t%.nc";

        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(head->get_output_port());
        w->set_file_name(fn);
        w->set_thread_pool_size(1);
        w->set_point_arrays({"q", "u", "v"});
        if (mask)
        {
            w->append_point_array("q_valid");
            w->append_point_array("u_valid");
            w->append_point_array("v_valid");
        }
        if (mask == 2)
        {
            w->append_point_array("zg");
        }
        w->set_executive(exec);

        w->update();
    }

    // compute IVT
    p_teca_integrated_vapor_transport ivt = teca_integrated_vapor_transport::New();
    ivt->set_input_connection(head->get_output_port());
    ivt->set_wind_u_variable("u");
    ivt->set_wind_v_variable("v");
    ivt->set_specific_humidity_variable("q");
    ivt->set_use_trapezoid_rule(trapezoid_int);

    if (trapezoid_int)
        std::cerr << "using trapezoid integrator" << std::endl;
    else
        std::cerr << "using first order integrator" << std::endl;


    // write the result
    if (write_output)
    {
        std::string fn = std::string("test_integrated_vapor_transport_output_") +
            std::string(mask ? "vv_mask" : "elev_mask") + std::string("_%t%.nc");

        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(ivt->get_output_port());
        w->set_executive(exec);
        w->set_file_name(fn);
        w->set_thread_pool_size(1);
        w->set_point_arrays({"ivt_u", "ivt_v"});
        w->update();
    }

    // capture the result
    p_teca_dataset_capture dsc = teca_dataset_capture::New();
    dsc->set_input_connection(ivt->get_output_port());
    dsc->set_executive(exec);
    dsc->update();

    const_p_teca_cartesian_mesh m =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(dsc->get_dataset());

    const_p_teca_double_array ivt_u =
        std::dynamic_pointer_cast<const teca_double_array>
            (m->get_point_arrays()->get("ivt_u"));

    const_p_teca_double_array ivt_v =
        std::dynamic_pointer_cast<const teca_double_array>
            (m->get_point_arrays()->get("ivt_v"));

    double test_ivt_u = ivt_u->get(0);
    double test_ivt_v = ivt_v->get(0);

    // calculate the analytic solution
    if (!trapezoid_int)
    {
        p_top = 0.0;
        p_sfc = 101325.0;
    }

    double base_ivt_u = -1./ (2. * a * g) * (
            sin(a * p_top)*sin(a * p_top)
          - sin(a * p_sfc)*sin(a * p_sfc)
        );

    double base_ivt_v =  1./ (a * g) * (
            (a * p_top * cos(a * p_top) - sin(a * p_top))
          - (a * p_sfc * cos(a * p_sfc) - sin(a * p_sfc))
        );


    // display
    std::cerr << "base_ivt_u = " << base_ivt_u << std::endl;
    std::cerr << "test_ivt_u = " << test_ivt_u << std::endl << std::endl;

    std::cerr << "base_ivt_v = " << base_ivt_v << std::endl;
    std::cerr << "test_ivt_v = " << test_ivt_v << std::endl;

    // compare against the analytic solution
    if (!teca_coordinate_util::equal(base_ivt_u, test_ivt_u, 1e-4))
    {
        TECA_ERROR("base_ivt_u = " << base_ivt_u << " test_ivt_u = " << test_ivt_u)
        return -1;
    }

    if (!teca_coordinate_util::equal(base_ivt_v, test_ivt_v, 1e-4))
    {
        TECA_ERROR("base_ivt_v = " << base_ivt_v << " test_ivt_v = " << test_ivt_v)
        return -1;
    }

    return 0;
}
