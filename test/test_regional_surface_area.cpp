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
#include "teca_shape_file_mask.h"

#include <cmath>
#include <functional>

using namespace teca_variant_array_util;


// the strategy of this test:
//
// given a flux that is everywhere 1 and a shape file, the result of the
// calculation is the surface area enclosed by the curve in the sshapefile
//

template <typename num_t>
struct ones
{
    p_teca_variant_array operator()(int, const const_p_teca_variant_array &lon,
        const const_p_teca_variant_array &lat, const const_p_teca_variant_array &plev,
        double t)
    {
        (void)plev;
        (void)t;

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
                    psflux[q] = NT(1);
                }
            }
            )

        return sflux;
    }
};

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 5)
    {
        std::cerr << "usage: test_surface_integral [grid res] [shape file] [area expected km2] [test tol]" << std::endl;
        return -1;
    }

    unsigned long lat_res = atoi(argv[1]);
    const char *shape_file = argv[2];
    double area_expect = atof(argv[3]);
    double test_tol = atof(argv[4]);

    p_teca_cartesian_mesh_source mesh = teca_cartesian_mesh_source::New();
    mesh->set_whole_extents({0, 2*lat_res, 0, lat_res, 0, 0, 0, 0});
    mesh->set_bounds({0., 360., -90., 90., 0., 0., 0., 0.});
    mesh->set_calendar("standard", "days since 2023-02-16 00:00:00");

    mesh->append_field_generator({"ones",
        teca_array_attributes(teca_variant_array_code<double>::get(),
            teca_array_attributes::point_centering, 0,
            teca_array_attributes::xyt_active(), "none", "ones",
            "a field of ones", 1, 1.e20),
            ones<double>{}});

    p_teca_shape_file_mask mask = teca_shape_file_mask::New();
    mask->set_input_connection(mesh->get_output_port());
    mask->set_shape_file(shape_file);
    mask->set_normalize_coordinates(1);
    mask->set_mask_variables({"reg_mask"});
    mask->set_verbose(1);

    // write the input
    bool write_input = false;
    if (write_input)
    {
        p_teca_cf_writer w = teca_cf_writer::New();
        w->set_input_connection(mask->get_output_port());
        w->set_file_name("reg_mask.nc");
        w->set_thread_pool_size(1);
        w->set_point_arrays({"ones", "reg_mask"});
        w->update();
        return 0;
    }

    // compute net surface flux
    p_teca_surface_integral mf = teca_surface_integral::New();
    mf->set_input_connection(mask->get_output_port());
    mf->set_input_variable("ones");
    mf->set_region_mask_variable("reg_mask");
    mf->set_output_variable("reg_area");

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
    auto col = tab->get_column_as<teca_double_array>("reg_area");
    double area = col->get(0) / 1.e6;
    double delta = fabs(area_expect - area);

    std::cerr
        << "res = " << 2*lat_res << ", " << lat_res << std::endl
        << "area km2 = " << area << std::endl
        << "area expect km2 = " << area_expect << std::endl
        << "delta = " << delta << std::endl
        << "test_tol = " << test_tol << std::endl;

    if (fabs(delta) > test_tol)
    {
        TECA_ERROR("Failed. Numerical error in flux calculation was larger than required")
        return -1;
    }

    return 0;
}
