
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_writer.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_coordinate_util.h"
#include "teca_dataset_capture.h"


#include <cmath>
#include <functional>


// ivt = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} q \vec{v} dp
//
// strategy: define integrands that are the product of 2 functions, subsitute in
// q, u, and v and evaluate numerically using our implementation and analytically
// and compare the results as a check
//
// let q = sin p , u = cos p, and v = p.
//
// we can then validate our implementation against the analyitic solutions:
//
// ivt_u = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} p sin p dp
//       = - \frac{1}{g} [ -p cos p + sin p + c ]
//
// ivt_v = - \frac{1}{g} \int_{p_{sfc}}^{p_{top}} cos p sin p dp
//       = - \frac{1}{g} [ \frac{1}{2} sin^2 p + c ]
//

template <typename num_t>
struct function_of_z
{
    using f_type = std::function<num_t(num_t)>;

    function_of_z(const f_type &a_f) : m_f(a_f) {}

    p_teca_variant_array operator()(const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
        double t)
    {
        (void)t;

        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();
        size_t nxy = nx*ny;

        p_teca_variant_array_impl<num_t> fz = teca_variant_array_impl<num_t>::New(nx*ny*nz);
        num_t *pfz = fz->get();

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            fz.get(),
            const NT *pz = dynamic_cast<const TT*>(z.get())->get();
            for (size_t k = 0; k < nz; ++k)
            {
                for (size_t j = 0; j < ny; ++j)
                {
                    for (size_t i = 0; i < nx; ++i)
                    {
                        pfz[k*nxy + j*nx + i] = this->m_f(pz[k]);
                    }
                }
            }
            )

        return fz;
    }

    f_type m_f;
};

// an actual sequence of pressure levels from the HighResMIP data
// plev = 92500, 85000, 70000, 60000, 50000, 25000, 5000 ;

int main(int argc, char **argv)
{
    unsigned long z1 = 1024;
    double p_sfc = 92500e-4;
    double p_top = 5000e-4;
    int write_input = 0;
    int write_output = 0;

    p_teca_cartesian_mesh_source s = teca_cartesian_mesh_source::New();
    s->set_whole_extents({0, 2, 0, 2, 0, z1, 0, 0});
    s->set_bounds({-1.0, 1.0, -1.0, 1.0, p_sfc, p_top, 0.0, 0.0});
    s->set_calendar("standard");
    s->set_time_units("days since 2020-09-30 00:00:00");

    // let q = sin(p)
    function_of_z<double> q([](double p) -> double { return sin(p); });

    s->append_field_generator({"q",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, "g kg^-1",
            "specific humidty", "test data where q = sin(p)"),
            q});

    // let u = cos(p)
    function_of_z<double> u([](double p) -> double { return cos(p); });

    s->append_field_generator({"u",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, "m s^-1",
            "longitudinal wind velocity", "test data where u = cos(p)"),
            u});

    // let v = p
    function_of_z<double> v([](double z) -> double { return z; });

    s->append_field_generator({"v",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0, "m s^-1",
            "latiitudinal wind velocity", "test data where v = p"),
            v});

    // write the test input dataset
    if (write_input)
    {
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(s->get_output_port());
        w->set_file_name("test_integrated_vapor_transport_input_%t%.nc");
        w->set_thread_pool_size(1);
        w->set_point_arrays({"q", "u", "v"});
        w->update();
    }

    // compute IVT
    p_teca_integrated_vapor_transport ivt = teca_integrated_vapor_transport::New();
    ivt->set_input_connection(s->get_output_port());
    ivt->set_wind_u_variable("u");
    ivt->set_wind_v_variable("v");
    ivt->set_specific_humidity_variable("q");

    // write the result
    if (write_output)
    {
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(ivt->get_output_port());
        w->set_file_name("test_integrated_vapor_transport_output_%t%.nc");
        w->set_thread_pool_size(1);
        w->set_point_arrays({"ivt_u", "ivt_v"});
        w->update();
    }

    // capture the result
    p_teca_dataset_capture dsc = teca_dataset_capture::New();
    dsc->set_input_connection(ivt->get_output_port());
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
    double g = 9.80665;

    double base_ivt_u = -(1./g) * (1./2.) * (sin(p_top)*sin(p_top)
                                           - sin(p_sfc)*sin(p_sfc));

    double base_ivt_v = -(1./g) * ((-p_top * cos(p_top) + sin(p_top))
                                 - (-p_sfc * cos(p_sfc) + sin(p_sfc)));

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
