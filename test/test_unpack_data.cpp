#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_valid_value_mask.h"
#include "teca_unpack_data.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_array_attributes.h"
#include "teca_index_executive.h"
#include "teca_system_util.h"
#include "teca_file_util.h"

#include "math.h"

using namespace teca_variant_array_util;


// compute data to pack
// f = cos(z)*sin(x+t)*sin(y+t)
// min(f) = -1
// max(f) = 1
struct packed_data
{
    unsigned char m_fill;
    float m_scale;
    float m_offset;

    packed_data()
    {
        // reserving 255 for the _FillValue
        // scale = (max(f) - min(f)) / (2^n - 2)
        // offs = min(f)
        m_scale = (1.0f - -1.0f)/254.0f;
        m_offset = -1.0f;
        m_fill = 255;
    }

    teca_metadata get_attributes()
    {
        teca_array_attributes aa(teca_variant_array_code<unsigned char>::get(),
           teca_array_attributes::point_centering,
           0, {1,1,1,1}, "unitless", "packed data", "cos(z)*sin(x+t)*sin(x+t)",
           1, m_fill);

        teca_metadata atts((teca_metadata)aa);

        atts.set("scale_factor", m_scale);
        atts.set("add_offset", m_offset);

        return atts;
    }

    p_teca_variant_array operator()(int,
        const const_p_teca_variant_array &x,
        const const_p_teca_variant_array &y,
        const const_p_teca_variant_array &z,
        double t)
    {
        size_t nx = x->size();
        size_t ny = y->size();
        size_t nz = z->size();

        size_t nxy = nx*ny;
        size_t nxyz = nxy*nz;

        // allocate f
        auto [f, p_f] = ::New<teca_float_array>(nxyz);

        // compute
        // f = cos(z)*sin(x+t)*sin(y+t)
        VARIANT_ARRAY_DISPATCH(x.get(),

            assert_type<CTT>(y, z);

            auto [sp_x, p_x,
                  sp_y, p_y,
                  sp_z, p_z] = get_host_accessible<CTT>(x, y, z);

            sync_host_access_any(x, y, z);

            for (size_t k = 0; k < nz; ++k)
            {
                for (size_t j = 0; j < ny; ++j)
                {
                    for (size_t i = 0; i < nx; ++i)
                    {
                        p_f[k*nxy + j*nx + i] = cos(p_z[k])*sin(p_x[i]+t)*sin(p_y[j]+t);
                    }
                }
            }
            )

        // allcate q
        auto [q, p_q] = ::New<teca_unsigned_char_array>(nxyz);

        // pack
        for (size_t i = 0; i < nxyz; ++i)
        {
            p_q[i] = (unsigned char)roundf((p_f[i] - m_offset)/m_scale);
        }

        // mask bottom and top row
        for (size_t i = 0; i < nx; ++i)
        {
            p_q[i] = m_fill;
            p_q[nxy - nx + i] = m_fill;
        }

        // mask left and right column
        for (size_t j = 0; j < ny; ++j)
        {
            p_q[j*nx] = m_fill;
            p_q[(j+1)*nx - 1] = m_fill;
        }

        return q;
    }
};


int main(int argc, char **argv)
{
    int write_input = 0;

    if (argc != 2)
    {
        std::cerr << "usage:" << std::endl
            << "test_unpack_data [baseline]" << std::endl;
        return -1;
    }

    std::string baseline = argv[1];

    packed_data pd;

    p_teca_cartesian_mesh_source src = teca_cartesian_mesh_source::New();
    src->set_coordinate_type_code(teca_variant_array_code<float>::get());
    src->set_field_type_code(teca_variant_array_code<unsigned char>::get());
    src->set_whole_extents({0, 63, 0, 63, 0, 0, 0, 15});
    src->set_bounds({-M_PI, M_PI, -M_PI, M_PI, 0.0, 0.0, 0.0, M_PI/4.});
    src->append_field_generator({"func", pd.get_attributes(), pd});
    src->set_calendar("standard", "days since 1980-01-01 00:00:00");

    if (write_input)
    {
        p_teca_cf_writer in_wri = teca_cf_writer::New();
        in_wri->set_input_connection(src->get_output_port());
        in_wri->set_point_arrays({"func"});
        in_wri->set_file_name("./test_unpack_data_input_%t%.nc");
        in_wri->set_thread_pool_size(1);
        in_wri->set_steps_per_file(64);
        in_wri->update();
    }

    p_teca_valid_value_mask vvm = teca_valid_value_mask::New();
    vvm->set_input_connection(src->get_output_port());

    p_teca_unpack_data unp = teca_unpack_data::New();
    unp->set_input_connection(vvm->get_output_port());

    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
    if (do_test && teca_file_util::file_exists(baseline.c_str()))
    {
        std::cerr << "running the test ... " << std::endl;

        p_teca_index_executive rex = teca_index_executive::New();
        rex->set_arrays({"func"});
        rex->set_verbose(1);

        p_teca_cf_reader rdr = teca_cf_reader::New();
        rdr->set_files_regex(baseline);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, rdr->get_output_port());
        diff->set_input_connection(1, unp->get_output_port());
        diff->set_executive(rex);

        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline ... " << std::endl;

        p_teca_cf_writer in_wri = teca_cf_writer::New();
        in_wri->set_input_connection(unp->get_output_port());
        in_wri->set_point_arrays({"func", "func_valid"});
        in_wri->set_file_name(baseline);
        in_wri->set_thread_pool_size(1);
        in_wri->set_steps_per_file(64);
        in_wri->update();

    }

    return 0;
}
