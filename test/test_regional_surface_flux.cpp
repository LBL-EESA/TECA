#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_writer.h"
#include "teca_valid_value_mask.h"
#include "teca_index_executive.h"
#include "teca_coordinate_util.h"
#include "teca_dataset_capture.h"
#include "teca_system_interface.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"
#include "teca_table_writer.h"
#include "teca_table_reduce.h"
#include "teca_surface_integral.h"
#include "teca_table.h"

#include <cmath>
#include <functional>

using namespace teca_variant_array_util;


// the strategy of this test:
//
// let the surface flux Q be given by: Q = 1/Re^2 cos theta
//
// then the net flux F over the square region S defined by [90, 270] degrees
// longitude and [-45, 45] degrees latitude is given by
//
// F = Re^2 \iint_{S} 1/Re^2 cos(theta) sin(phi) dtheta dphi
//   = 2 sqrt(2)
//


// generate the region mask
//
//        / 1 where lon \in [90, 270] and lat \in [-45, 45]
// mask = |
//        \ 0 every where else
//
template <typename num_t>
struct region_mask
{
    p_teca_variant_array operator()(int, const const_p_teca_variant_array &lon,
        const const_p_teca_variant_array &lat, const const_p_teca_variant_array &plev,
        double t)
    {
        (void)plev;
        (void)t;

        size_t nlon = lon->size();
        size_t nlat = lat->size();

        auto [reg_mask, preg_mask] = New<teca_variant_array_impl<num_t>>(nlon*nlat);

        VARIANT_ARRAY_DISPATCH(lon.get(),

            auto [splon, plon, splat, plat] = get_host_accessible<CTT>(lon, lat);
            sync_host_access_any(lon, lat);

            for (size_t j = 0; j < nlat; ++j)
            {
                for (size_t i = 0; i < nlon; ++i)
                {
                    size_t q = j*nlon + i;

                    NT lon_i = plon[i];
                    NT lat_j = plat[j];

                    if (((lon_i >= 90.) && (lon_i <= 270.)) && ((lat_j >=-45.) && (lat_j <= 45.)))
                    {
                        preg_mask[q] = num_t(1);
                    }
                    else
                    {
                        preg_mask[q] = num_t(0);
                    }
                }
            }
            )

        return reg_mask;
    }
};

template <typename num_t>
struct surface_flux
{
    p_teca_variant_array operator()(int, const const_p_teca_variant_array &lon,
        const const_p_teca_variant_array &lat, const const_p_teca_variant_array &plev,
        double t)
    {
        (void)plev;
        (void)t;

        num_t Re2_inv = 1. / (6378100. * 6378100.);
        num_t rad_deg = M_PI / 180.;

        size_t nlon = lon->size();
        size_t nlat = lat->size();
        size_t nelem = nlon*nlat;

        auto [sflux, psflux] = New<teca_variant_array_impl<num_t>>(nelem);

        VARIANT_ARRAY_DISPATCH(lon.get(),

            auto [splon, plon] = get_host_accessible<CTT>(lon);
            sync_host_access_any(lon);

            for (size_t j = 0; j < nlat; ++j)
            {
                for (size_t i = 0; i < nlon; ++i)
                {
                    size_t q = j*nlon + i;

                    NT theta_i = plon[i] * rad_deg;

                    psflux[q] = Re2_inv * cos( theta_i );
                }
            }
            )

        return sflux;
    }
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 3)
    {
        std::cerr << "usage: test_surface_integral [grid res] [test tol]" << std::endl;
        return -1;
    }

    unsigned long lat_res = atoi(argv[1]);
    double test_tol = atof(argv[2]);

    p_teca_cartesian_mesh_source mesh = teca_cartesian_mesh_source::New();
    mesh->set_whole_extents({0, 2*lat_res, 0, lat_res, 0, 0, 0, 0});
    mesh->set_bounds({0., 360., -90., 90., 0., 0., 0., 0.});
    mesh->set_calendar("standard", "days since 2023-02-16 00:00:00");

    mesh->append_field_generator({"reg_mask",
        teca_array_attributes(teca_variant_array_code<char>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyt_active(), "unitless", "region_mask",
            "mask is 1 where lon in [90, 270] and lat in [-45, 45]", 1, char(254)),
            region_mask<char>{}});

    mesh->append_field_generator({"flux",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyt_active(), "kg m-2 s-1", "mass flux",
            "flux = 1/Re^2 * cos( lon )", 1, 1.e20),
            surface_flux<double>{}});


    // write the input
    bool write_input = false;
    if (write_input)
    {
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(mesh->get_output_port());
        w->set_file_name("test_inputs_%t%.nc");
        w->set_thread_pool_size(1);
        w->set_point_arrays({"flux", "reg_mask"});
        w->update();
        return 0;
    }

    // compute net surface flux
    p_teca_surface_integral mf = teca_surface_integral::New();
    mf->set_input_connection(mesh->get_output_port());
    mf->set_input_variable("flux");
    mf->set_region_mask_variable("reg_mask");
    mf->set_output_prefix("net_");

    p_teca_table_reduce mr = teca_table_reduce::New();
    mr->set_input_connection(mf->get_output_port());
    mr->set_thread_pool_size(1);

    bool write_output = false;
    if (write_output)
    {
        auto twr = teca_table_writer::New();
        twr->set_input_connection(mr->get_output_port());
        twr->set_file_name("test_output.csv");
        twr->update();
        return 0;
    }

    // capture the result
    p_teca_dataset_capture dsc = teca_dataset_capture::New();
    dsc->set_input_connection(mr->get_output_port());
    //dsc->set_executive(exec);
    dsc->update();

    auto tab = dsc->get_dataset_as<teca_table>();
    auto col = tab->get_column_as<teca_double_array>("net_flux");
    double flux = col->get(0);
    double exact = -2.0*sqrt(2.0);
    double delta = fabs(exact - flux);

    std::cerr
        << "res = " << 2*lat_res << ", " << lat_res << std::endl
        << "flux = " << flux << std::endl
        << "exact = " << exact << std::endl
        << "delta = " << delta << std::endl
        << "test_tol = " << test_tol << std::endl;

    if (fabs(delta) > test_tol)
    {
        TECA_ERROR("Failed. Numerical error in flux calculation was larger than required")
        return -1;
    }

    return 0;
}
