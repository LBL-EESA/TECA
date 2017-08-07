#include "teca_cartesian_mesh_source.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_coordinate_util.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

struct teca_cartesian_mesh_source::internals_t
{
    teca_metadata metadata;
};

// --------------------------------------------------------------------------
teca_cartesian_mesh_source::teca_cartesian_mesh_source() :
    coordinate_type_code(1/*teca_variant_array_code<float>::get()*/),
    internals(new internals_t)
{
    this->set_number_of_input_connections(0);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_source::~teca_cartesian_mesh_source()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    (void)prefix;
    (void)global_opts;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_properties(const std::string &prefix,
    variables_map &opts)
{
    (void)prefix;
    (void)opts;
}
#endif


template<typename num_t>
void initialize_axis(p_teca_variant_array_impl<num_t> x,
    unsigned long i0, unsigned long i1, num_t x0, num_t x1)
{
    unsigned long nx = i1 - i0 + 1;
    num_t dx = (x1 - x0)/(nx - 1l);
    num_t xx = x0 + i0*dx;
    x->resize(nx);
    num_t *px = x->get();
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = xx + dx*i;
}


void initialize_axes(int type_code, unsigned long *extent,
    double *bounds, p_teca_variant_array &x_axis,
    p_teca_variant_array &y_axis, p_teca_variant_array &z_axis)
{
    x_axis = teca_variant_array_factory::New(type_code);
    y_axis = x_axis->new_instance();
    z_axis = x_axis->new_instance();

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        x_axis.get(),

        initialize_axis<NT>(std::static_pointer_cast<TT>(x_axis),
            extent[0], extent[1], bounds[0], bounds[1]);

        initialize_axis<NT>(std::static_pointer_cast<TT>(y_axis),
            extent[2], extent[3], bounds[2], bounds[3]);

        initialize_axis<NT>(std::static_pointer_cast<TT>(z_axis),
            extent[4], extent[5], bounds[4], bounds[5]);
        )
}


// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::clear_cached_metadata()
{
    this->internals->metadata.clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_source::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_source::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;
    if (this->internals->metadata)
        return this->internals->metadata;

    if (this->whole_extents.size() != 6)
    {
        TECA_ERROR("invalid whole extents were specified")
        return teca_metadata();
    }

    if (this->bounds.size() != 6)
    {
        TECA_ERROR("invalid bounds were specified")
        return teca_metadata();
    }

    p_teca_variant_array x_axis, y_axis, z_axis;

    initialize_axes(this->coordinate_type_code, this->whole_extents.data(),
        this->bounds.data(), x_axis, y_axis, z_axis);

    teca_metadata coords;
    coords.set("x_variable", (this->x_axis_variable.empty() ? "x" : this->x_axis_variable));
    coords.set("y_variable", (this->y_axis_variable.empty() ? "y" : this->y_axis_variable));
    coords.set("z_variable", (this->z_axis_variable.empty() ? "z" : this->z_axis_variable));
    coords.set("t_variable", (this->t_axis_variable.empty() ? "t" : this->t_axis_variable));

    coords.set("x", x_axis);
    coords.set("y", y_axis);
    coords.set("z", z_axis);

    p_teca_double_array t_axis;
    coords.set("t", t_axis);

    this->internals->metadata.set("whole_extent", this->whole_extents);
    this->internals->metadata.set("coordinates", coords);

    return this->internals->metadata;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_source::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_source::execute" << endl;
#endif
    (void)port;
    (void)input_data;
    (void)request;

    // get coordinates
    teca_metadata coords;
    if (this->internals->metadata.get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing \"coordinates\"")
        return nullptr;
    }

    p_teca_variant_array in_x, in_y, in_z, in_t;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")) || !(in_t = coords.get("t")))
    {
        TECA_ERROR("metadata is missing coordinate arrays")
        return nullptr;
    }

    unsigned long md_whole_extent[6] = {0};
    if (this->internals->metadata.get("whole_extent", md_whole_extent, 6))
    {
        TECA_ERROR("metadata is missing \"whole_extent\"")
        return nullptr;
    }

    unsigned long req_extent[6] = {0};
    double req_bounds[6] = {0.0};
    if (request.get("bounds", req_bounds, 6))
    {
        // bounds key not present, check for extent key
        // if not present use whole_extent
        if (request.get("extent", req_extent, 6))
        {
            memcpy(req_extent, md_whole_extent, 6*sizeof(unsigned long));
        }
    }
    else
    {
        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        if (teca_coordinate_util::bounds_to_extent(
            req_bounds, in_x, in_y, in_z, req_extent))
        {
            TECA_ERROR("invalid bounds requested.")
            return nullptr;
        }
    }

    // slice axes on the requested extent
    p_teca_variant_array out_x = in_x->new_copy(req_extent[0], req_extent[1]);
    p_teca_variant_array out_y = in_y->new_copy(req_extent[2], req_extent[3]);
    p_teca_variant_array out_z = in_z->new_copy(req_extent[4], req_extent[5]);

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates(out_x);
    mesh->set_y_coordinates(out_y);
    mesh->set_z_coordinates(out_z);
    mesh->set_whole_extent(md_whole_extent);
    mesh->set_extent(req_extent);

    return mesh;
}
