#include "teca_cartesian_mesh_source.h"
#include "teca_programmable_algorithm.h"
#include "teca_cf_writer.h"
#include "teca_array_attributes.h"
#include "teca_index_executive.h"


// this function returns a double array initialized with time
// values.
p_teca_variant_array generate_mesh_time(const const_p_teca_variant_array &x,
    const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
    double t)
{
    size_t nx = x->size();
    size_t ny = y->size();
    size_t nz = z->size();

    size_t nxyz = nx*ny*nz;

    p_teca_double_array da = teca_double_array::New(nxyz);

    double *pda = da->get();

    for (size_t i = 0; i < nxyz; ++i)
        pda[i] = t;

    return da;
}

int main(int, char **)
{
    // this test intentionally declares a mesh array using the
    // wrong type code. this will test the error handling feature
    // in the cf_weriter.
    p_teca_cartesian_mesh_source ms = teca_cartesian_mesh_source::New();
    ms->set_whole_extents({0, 359, 0, 179, 0, 0, 0, 7});
    ms->set_bounds({0.0, 360.0, -90.0, 90.0, 0.0, 0.0, 0.0, 10.0});
    ms->set_calendar("standard", "days since 07-14-2020");
    ms->append_field_generator({"mesh_time",
        teca_array_attributes(teca_variant_array_code<int>::get(), // this is the wrong type code!
            teca_array_attributes::point_centering, 0, "days since 01-01-1980",
            "mesh time values", "a mesh sized array filled in with the current time"),
        generate_mesh_time});

    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_verbose(1);

    p_teca_cf_writer w = teca_cf_writer::New();
    w->set_input_connection(ms->get_output_port());
    w->set_verbose(1);
    w->set_thread_pool_size(1);
    w->set_file_name("test_bad_type_%t%.nc");
    w->set_steps_per_file(32);
    w->set_point_arrays({"mesh_time"});
    w->set_executive(exec);

    w->update();

    return 0;
}
