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
    // given the index space i0 to i1 spanning the world space
    // x0 to x1, generate an equally spaced coordinate axis
    template<typename num_t>
    static
    void initialize_axis(p_teca_variant_array_impl<num_t> x,
        unsigned long i0, unsigned long i1, num_t x0, num_t x1);

    // given an index space [i0 i1 j0 j1 k0 k1 q0 q1] spanning
    // the world space [x0 x1 y0 y1 z0 z1 t0 t1] generate
    // equally spaced coordinate axes x,y,z,t
    static
    void initialize_axes(int type_code, unsigned long *extent,
        double *bounds, p_teca_variant_array &x_axis,
        p_teca_variant_array &y_axis, p_teca_variant_array &z_axis,
        p_teca_variant_array &t_axis);

    teca_metadata metadata;
};


// --------------------------------------------------------------------------
template<typename num_t>
void teca_cartesian_mesh_source::internals_t::initialize_axis(
    p_teca_variant_array_impl<num_t> x, unsigned long i0, unsigned long i1,
    num_t x0, num_t x1)
{
    // generate an equally spaced coordinate axes
    unsigned long nx = i1 - i0 + 1;
    x->resize(nx);

    // avoid divide by zero
    if (nx < 2)
        return;

    num_t dx = (x1 - x0)/(nx - 1l);
    num_t xx = x0 + i0*dx;

    num_t *px = x->get();
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = xx + dx*i;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::internals_t::initialize_axes(int type_code,
    unsigned long *extent, double *bounds, p_teca_variant_array &x_axis,
    p_teca_variant_array &y_axis, p_teca_variant_array &z_axis,
    p_teca_variant_array &t_axis)
{
    // gernate equally spaced coordinate axes x,y,z,t
    x_axis = teca_variant_array_factory::New(type_code);
    y_axis = x_axis->new_instance();
    z_axis = x_axis->new_instance();
    t_axis = x_axis->new_instance();

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        x_axis.get(),

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(x_axis),
            extent[0], extent[1], bounds[0], bounds[1]);

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(y_axis),
            extent[2], extent[3], bounds[2], bounds[3]);

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(z_axis),
            extent[4], extent[5], bounds[4], bounds[5]);

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(t_axis),
            extent[6], extent[7], bounds[6], bounds[7]);
        )
}



// --------------------------------------------------------------------------
teca_cartesian_mesh_source::teca_cartesian_mesh_source() :
    coordinate_type_code(teca_variant_array_code<double>::get()),
    field_type_code(teca_variant_array_code<double>::get()),
    x_axis_variable("lon"), y_axis_variable("lat"), z_axis_variable("plev"),
    t_axis_variable("time"), x_axis_units("degrees_east"),
    y_axis_units("degrees_north"), z_axis_units("pascals"),
    calendar("Gregorian"), time_units("seconds since 1970-01-01 00:00:00"),
    whole_extents{0l, 359l, 0l, 179l, 0l, 0l, 0l, 0l},
    bounds{0., 360, -90., 90., 0., 0., 0., 0.},
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
void teca_cartesian_mesh_source::append_field_generator(
    const std::string &name, field_generator_callback &callback)
{
    this->append_field_generator({name, callback});
}

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_source::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_source::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;
    if (this->internals->metadata)
        return this->internals->metadata;

    if (this->whole_extents.size() != 8)
    {
        TECA_ERROR("invalid whole extents were specified")
        return teca_metadata();
    }

    if (this->bounds.size() != 8)
    {
        TECA_ERROR("invalid bounds were specified")
        return teca_metadata();
    }

    // generate cooridnate axes
    p_teca_variant_array x_axis, y_axis, z_axis, t_axis;

    internals_t::initialize_axes(this->coordinate_type_code,
        this->whole_extents.data(), this->bounds.data(), x_axis,
        y_axis, z_axis, t_axis);

    std::string x_ax_var_name = (this->x_axis_variable.empty() ? "x" : this->x_axis_variable);
    std::string y_ax_var_name = (this->y_axis_variable.empty() ? "y" : this->y_axis_variable);
    std::string z_ax_var_name = (this->z_axis_variable.empty() ? "z" : this->z_axis_variable);
    std::string t_ax_var_name = (this->t_axis_variable.empty() ? "t" : this->t_axis_variable);

    // construct attributes
    teca_metadata x_atts;
    x_atts.set("units", (this->x_axis_units.empty() ? "meters" : this->x_axis_units));

    teca_metadata y_atts;
    y_atts.set("units", (this->y_axis_units.empty() ? "meters" : this->y_axis_units));

    teca_metadata z_atts;
    z_atts.set("units", (this->z_axis_units.empty() ? "meters" : this->z_axis_units));

    teca_metadata t_atts;
    t_atts.set("units", (this->time_units.empty() ?
        "seconds since 1970-01-01 00:00:00" : this->time_units));

    teca_metadata atts;
    atts.set(x_ax_var_name, x_atts);
    atts.set(y_ax_var_name, y_atts);
    atts.set(z_ax_var_name, z_atts);
    atts.set(t_ax_var_name, t_atts);

    // construct dataset metadata
    teca_metadata coords;
    coords.set("x_variable", x_ax_var_name);
    coords.set("y_variable", y_ax_var_name);
    coords.set("z_variable", z_ax_var_name);
    coords.set("t_variable", t_ax_var_name);

    coords.set("x", x_axis);
    coords.set("y", y_axis);
    coords.set("z", z_axis);
    coords.set("t", t_axis);

    this->internals->metadata.set("whole_extent", this->whole_extents);
    this->internals->metadata.set("coordinates", coords);

    std::vector<std::string> vars;
    std::vector<field_generator_t>::iterator it = this->field_generators.begin();
    std::vector<field_generator_t>::iterator end = this->field_generators.end();
    for (; it != end; ++it)
        vars.push_back(it->name);

    this->internals->metadata.set("variables", vars);
    this->internals->metadata.set("attributes", atts);

    this->internals->metadata.set("number_of_time_steps", t_axis->size());
    this->internals->metadata.set("index_initializer_key", std::string("number_of_time_steps"));
    this->internals->metadata.set("index_request_key", std::string("time_step"));

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

    // get the extent of the dataset we could generate
    unsigned long md_whole_extent[6] = {0};
    if (this->internals->metadata.get("whole_extent", md_whole_extent, 6))
    {
        TECA_ERROR("metadata is missing \"whole_extent\"")
        return nullptr;
    }

    // determine the subset we are being requested, fall back to the
    // whole extent if a subset isn't specified
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

    // get the timestep
    unsigned long time_step = 0;
    if (request.get("time_step", time_step))
    {
        TECA_ERROR("Request is missing time_step")
        return nullptr;
    }

    // get the time
    double t = 0.;
    in_t->get(time_step, t);

    // slice axes on the requested extent
    p_teca_variant_array out_x = in_x->new_copy(req_extent[0], req_extent[1]);
    p_teca_variant_array out_y = in_y->new_copy(req_extent[2], req_extent[3]);
    p_teca_variant_array out_z = in_z->new_copy(req_extent[4], req_extent[5]);

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();

    std::string x_variable = this->x_axis_variable.empty() ? "x" : this->x_axis_variable;
    std::string y_variable = this->y_axis_variable.empty() ? "y" : this->y_axis_variable;
    std::string z_variable = this->z_axis_variable.empty() ? "z" : this->z_axis_variable;
    std::string t_variable = this->t_axis_variable.empty() ? "t" : this->t_axis_variable;

    mesh->set_x_coordinates(x_variable, out_x);
    mesh->set_y_coordinates(y_variable, out_y);
    mesh->set_z_coordinates(z_variable, out_z);

    // set metadata
    mesh->set_whole_extent(md_whole_extent);
    mesh->set_extent(req_extent);
    mesh->set_time_step(time_step);
    mesh->set_time(t);
    mesh->set_calendar(this->calendar);
    mesh->set_time_units(this->time_units);

    teca_metadata &mesh_md = mesh->get_metadata();
    mesh_md.set("index_request_key", std::string("time_step"));

    // generate fields over the requested subset
    std::vector<field_generator_t>::iterator it = this->field_generators.begin();
    std::vector<field_generator_t>::iterator end = this->field_generators.end();
    for (; it != end; ++it)
    {
        p_teca_variant_array f_xyzt = it->generator(out_x, out_y, out_z, t);
        mesh->get_point_arrays()->append(it->name, f_xyzt);
    }

    // pass the attributes
    teca_metadata atts;
    this->internals->metadata.get("attributes", atts);
    mesh_md.set("attributes", atts);

    return mesh;
}
