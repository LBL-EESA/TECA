#include "teca_cartesian_mesh_source.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_programmable_algorithm.h"
#include "teca_cf_writer.h"
#include "teca_array_attributes.h"
#include "teca_index_executive.h"

using namespace teca_variant_array_util;

// this function returns a double array initialized with time
// values.
p_teca_variant_array generate_mesh_time(int,
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, double t)
{
    size_t nx = x->size();
    size_t ny = y->size();
    size_t nz = z->size();

    size_t nxyz = nx*ny*nz;

    auto [da, pda] = ::New<teca_double_array>(nxyz);

    for (size_t i = 0; i < nxyz; ++i)
        pda[i] = t;

    return da;
}


TECA_SHARED_OBJECT_FORWARD_DECL(change_array_data_type_to_double)

// this class changes the data type of the "mesh_time" variable to double
class change_array_data_type_to_double : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(change_array_data_type_to_double)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(change_array_data_type_to_double)
    TECA_ALGORITHM_CLASS_NAME(change_array_data_type_to_double)
    ~change_array_data_type_to_double() {}

protected:

    change_array_data_type_to_double()
    {
        this->set_number_of_input_connections(1);
        this->set_number_of_output_ports(1);
    }

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override
    {
        (void)port;
        (void)request;

        const_p_teca_cartesian_mesh cm =
            std::static_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

        const_p_teca_array_collection pa = cm->get_point_arrays();

        p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
        out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(cm));

        p_teca_double_array da = teca_double_array::New();
        da->copy(pa->get("mesh_time"));

        out_mesh->get_point_arrays()->set("mesh_time", da);

        return out_mesh;
    }
};


int main(int, char **)
{
    // set the error handler to print and return
    teca_error::set_error_message_handler();

    // this test intentionally declares a mesh array using the
    // wrong type code. this will test the error handling feature
    // in the cf_writer.
    p_teca_cartesian_mesh_source ms = teca_cartesian_mesh_source::New();
    ms->set_whole_extents({0, 359, 0, 179, 0, 0, 0, 7});
    ms->set_bounds({0.0, 360.0, -90.0, 90.0, 0.0, 0.0, 0.0, 10.0});
    ms->set_calendar("standard", "days since 07-14-2020");
    ms->append_field_generator({"mesh_time",
        teca_array_attributes(teca_variant_array_code<int>::get(), // this is the wrong type code!
            teca_array_attributes::point_centering, 0, {1,1,0,1}, "days since 01-01-1980",
            "mesh time values", "a mesh sized array filled in with the current time"),
        generate_mesh_time});

    p_change_array_data_type_to_double ct
        = change_array_data_type_to_double::New();

    ct->set_input_connection(ms->get_output_port());

    p_teca_index_executive exec = teca_index_executive::New();
    exec->set_verbose(1);

    p_teca_cf_writer w = teca_cf_writer::New();
    w->set_input_connection(ct->get_output_port());
    w->set_verbose(1);
    w->set_thread_pool_size(1);
    w->set_file_name("test_bad_type_%t%.nc");
    w->set_steps_per_file(32);
    w->set_point_arrays({"mesh_time"});
    w->set_executive(exec);

    w->update();

    return 0;
}
