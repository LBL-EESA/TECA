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
    void initialize_axes(int type_code, const unsigned long *extent,
        const double *bounds, p_teca_variant_array &x_axis,
        p_teca_variant_array &y_axis, p_teca_variant_array &z_axis,
        p_teca_variant_array &t_axis);

    static
    void initialize_axes(int type_code, const unsigned long *extent,
        const double *bounds, p_teca_variant_array &x_axis,
        p_teca_variant_array &y_axis, p_teca_variant_array &z_axis);

    // cached metadata
    teca_metadata metadata;
    p_teca_variant_array t_axis;
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

    num_t *px = x->get();

    // avoid divide by zero
    if (nx < 2)
    {
        px[0] = x0;
        return;
    }

    num_t dx = (x1 - x0)/(nx - 1l);
    num_t xx = x0 + i0*dx;

    for (unsigned long i = 0; i < nx; ++i)
        px[i] = xx + dx*i;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::internals_t::initialize_axes(int type_code,
    const unsigned long *extent, const double *bounds, p_teca_variant_array &x_axis,
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
void teca_cartesian_mesh_source::internals_t::initialize_axes(int type_code,
    const unsigned long *extent, const double *bounds, p_teca_variant_array &x_axis,
    p_teca_variant_array &y_axis, p_teca_variant_array &z_axis)
{
    // gernate equally spaced coordinate axes x,y,z,t
    x_axis = teca_variant_array_factory::New(type_code);
    y_axis = x_axis->new_instance();
    z_axis = x_axis->new_instance();

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        x_axis.get(),

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(x_axis),
            extent[0], extent[1], bounds[0], bounds[1]);

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(y_axis),
            extent[2], extent[3], bounds[2], bounds[3]);

        internals_t::initialize_axis<NT>(std::static_pointer_cast<TT>(z_axis),
            extent[4], extent[5], bounds[4], bounds[5]);
        )
}



// --------------------------------------------------------------------------
teca_cartesian_mesh_source::teca_cartesian_mesh_source() :
    coordinate_type_code(teca_variant_array_code<double>::get()),
    field_type_code(teca_variant_array_code<double>::get()),
    x_axis_variable("lon"), y_axis_variable("lat"), z_axis_variable("plev"),
    t_axis_variable("time"),
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
    teca_algorithm::set_modified();
}



// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_x_axis_variable(const std::string &name)
{
    this->x_axis_variable = name;
    this->x_axis_attributes.clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_x_axis_variable(const std::string &name,
    const teca_metadata &atts)
{
    this->x_axis_variable = name;
    this->x_axis_attributes = atts;
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_x_axis_variable(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    if (coords.get("x_variable", this->x_axis_variable))
        return -1;

    teca_metadata atts;
    if (md.get("attributes", atts))
        return -1;

    if (atts.get(this->x_axis_variable, this->x_axis_attributes))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_y_axis_variable(const std::string &name)
{
    this->y_axis_variable = name;
    this->y_axis_attributes.clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_y_axis_variable(const std::string &name,
    const teca_metadata &atts)
{
    this->y_axis_variable = name;
    this->y_axis_attributes = atts;
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_y_axis_variable(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    if (coords.get("y_variable", this->y_axis_variable))
        return -1;

    teca_metadata atts;
    if (md.get("attributes", atts))
        return -1;

    if (atts.get(this->y_axis_variable, this->y_axis_attributes))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_z_axis_variable(const std::string &name)
{
    this->z_axis_variable = name;
    this->z_axis_attributes.clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_z_axis_variable(const std::string &name,
    const teca_metadata &atts)
{
    this->z_axis_variable = name;
    this->z_axis_attributes = atts;
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_z_axis_variable(const teca_metadata &md)
{
    // get coordinates and attributes, fail if either are missing
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    if (coords.get("x_variable", this->z_axis_variable))
        return -1;

    teca_metadata atts;
    if (md.get("attributes", atts))
        return -1;

    if (atts.get(this->z_axis_variable, this->z_axis_attributes))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_t_axis_variable(const std::string &name)
{
    this->t_axis_variable = name;
    this->t_axis_attributes.clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_calendar(
    const std::string &calendar, const std::string &units)
{
    this->t_axis_attributes.clear();
    this->t_axis_attributes.set("calendar", calendar);
    this->t_axis_attributes.set("units", units);
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_t_axis_variable(const std::string &name,
    const teca_metadata &atts)
{
    this->t_axis_variable = name;
    this->t_axis_attributes = atts;
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_t_axis_variable(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    if (coords.get("t_variable", this->t_axis_variable))
        return -1;

    teca_metadata atts;
    if (md.get("attributes", atts))
        return -1;

    if (atts.get(this->t_axis_variable, this->t_axis_attributes))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_t_axis(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    this->internals->t_axis = coords.get("t");

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::set_t_axis(const p_teca_variant_array &t)
{
    this->internals->t_axis = t;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_output_metadata(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    teca_metadata atts;
    if (md.get("attributes", atts))
        return -1;

    // get the coordinate axes.
    const_p_teca_variant_array x = coords.get("x");
    const_p_teca_variant_array y = coords.get("y");
    const_p_teca_variant_array z = coords.get("z");
    const_p_teca_variant_array t = coords.get("t");

    // because of assumptions made in execute, all must be provided
    if (!x || !y || !z || !t)
        return -1;

    unsigned long nx = x->size();
    unsigned long ny = y->size();
    unsigned long nz = z->size();
    unsigned long nxyz = nx*ny*nz;

    // clear out any variables, and replace with those that we provide.
    std::vector<std::string> vars;
    std::vector<field_generator_t>::iterator it = this->field_generators.begin();
    std::vector<field_generator_t>::iterator end = this->field_generators.end();
    for (; it != end; ++it)
    {
        vars.push_back(it->name);

        // correct size
        teca_metadata var_atts = it->attributes;
        var_atts.set("size", nxyz);

        atts.set(it->name, var_atts);
    }

    // copy the metadata
    this->set_modified();

    this->internals->metadata = md;
    this->internals->metadata.set("variables", vars);

    return 0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_spatial_extents(const teca_metadata &md,
    bool three_d)
{
    // get coordinates and attributes, fail if either are missing
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    teca_metadata attributes;
    if (md.get("attributes", attributes))
        return -1;

    // get the coordinate axes
    p_teca_variant_array x = coords.get("x");
    p_teca_variant_array y = coords.get("y");
    p_teca_variant_array z = coords.get("z");

    // verify
    if (!x || !y || (three_d && !z))
        return -1;

    // set the extents
    this->whole_extents[0] = 0;
    this->whole_extents[1] = x->size() - 1;
    this->whole_extents[2] = 0;
    this->whole_extents[3] = y->size() - 1;
    this->whole_extents[4] = 0;
    this->whole_extents[5] = three_d ? z->size() - 1 : 0;

    return 0;
}
// --------------------------------------------------------------------------
int teca_cartesian_mesh_source::set_spatial_bounds(const teca_metadata &md,
    bool three_d)
{
    // get coordinates and attributes, fail if either are missing
    teca_metadata coords;
    if (md.get("coordinates", coords))
        return -1;

    teca_metadata attributes;
    if (md.get("attributes", attributes))
        return -1;

    // get the coordinate axes
    p_teca_variant_array x = coords.get("x");
    p_teca_variant_array y = coords.get("y");
    p_teca_variant_array z = coords.get("z");

    // verify
    if (!x || !y || (three_d && !z))
        return -1;

    // get the bounds
    x->get(0lu, this->bounds[0]);
    x->get(x->size() - 1lu, this->bounds[1]);
    y->get(0lu, this->bounds[2]);
    y->get(y->size() - 1lu, this->bounds[3]);
    z->get(0lu, this->bounds[4]);

    unsigned long khi = three_d ? z->size() - 1lu : 0lu;
    z->get(khi, this->bounds[5]);

    // set the coordinate type
    this->set_coordinate_type_code(x->type_code());

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_source::append_field_generator(
    const std::string &name, const teca_metadata &atts,
    field_generator_callback &callback)
{
    this->append_field_generator({name, atts, callback});
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

    if (this->internals->t_axis)
    {
        // generate x,y,z axes but use cached time axis
        internals_t::initialize_axes(this->coordinate_type_code,
            this->whole_extents.data(), this->bounds.data(), x_axis,
            y_axis, z_axis);

        t_axis = this->internals->t_axis;
    }
    else
    {
        // generate x,y,z and t axes
        internals_t::initialize_axes(this->coordinate_type_code,
            this->whole_extents.data(), this->bounds.data(), x_axis,
            y_axis, z_axis, t_axis);
    }

    size_t nx = x_axis->size();
    size_t ny = y_axis->size();
    size_t nz = z_axis->size();
    size_t nt = t_axis->size();

    // construct attributes
    teca_metadata x_atts = this->x_axis_attributes;
    x_atts.set("type_code", x_axis->type_code());
    x_atts.set("size", nx);

    teca_metadata y_atts = this->y_axis_attributes;
    y_atts.set("type_code", y_axis->type_code());
    y_atts.set("size", ny);

    teca_metadata z_atts = this->z_axis_attributes;
    z_atts.set("type_code", z_axis->type_code());
    z_atts.set("size", nz);

    teca_metadata t_atts = this->t_axis_attributes;
    t_atts.set("type_code", t_axis->type_code());
    t_atts.set("size", nt);

    teca_metadata atts;
    atts.set(this->x_axis_variable, x_atts);
    atts.set(this->y_axis_variable, y_atts);
    atts.set(this->z_axis_variable, z_atts);

    if (!this->t_axis_variable.empty())
        atts.set(this->t_axis_variable, t_atts);

    // construct dataset metadata
    teca_metadata coords;
    coords.set("x_variable", this->x_axis_variable);
    coords.set("y_variable", this->y_axis_variable);
    coords.set("z_variable", this->z_axis_variable);
    coords.set("t_variable", this->t_axis_variable);

    coords.set("x", x_axis);
    coords.set("y", y_axis);
    coords.set("z", z_axis);
    coords.set("t", t_axis);

    this->internals->metadata.set("whole_extent", this->whole_extents);
    this->internals->metadata.set("coordinates", coords);

    size_t nxyz = nx*ny*nz;
    std::vector<std::string> vars;
    std::vector<field_generator_t>::iterator it = this->field_generators.begin();
    std::vector<field_generator_t>::iterator end = this->field_generators.end();
    for (; it != end; ++it)
    {
        vars.push_back(it->name);

        // correct size
        teca_metadata var_atts = it->attributes;
        var_atts.set("size", nxyz);

        atts.set(it->name, var_atts);
    }

    this->internals->metadata.set("variables", vars);
    this->internals->metadata.set("attributes", atts);

    // setup the execution control keys
    this->internals->metadata.set("number_of_time_steps",
        t_axis->size());

    this->internals->metadata.set("index_initializer_key",
        std::string("number_of_time_steps"));

    this->internals->metadata.set("index_request_key",
        std::string("time_step"));

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
        if (teca_coordinate_util::bounds_to_extent(req_bounds,
                in_x, in_y, in_z, req_extent) ||
            teca_coordinate_util::validate_extent(in_x->size(),
                in_y->size(), in_z->size(), req_extent, true))
        {
            TECA_ERROR("invalid bounds requested.")
            return nullptr;
        }
    }

    // get the timestep, no matter what the key is named we treat it as
    // a time step. this is to support metadata provided by another source
    // eg. a different reader.
    std::string request_key;
    if (request.get("index_request_key", request_key))
    {
        TECA_ERROR("Request is missing the \"index_request_key\"")
        return nullptr;
    }

    unsigned long req_index = 0;
    if (request.get(request_key, req_index))
    {
        TECA_ERROR("Request is missing \"" << request_key << "\"")
        return nullptr;
    }

    // check that the we have a time value for the requested index.
    if (req_index >= in_t->size())
    {
        TECA_ERROR("The requested index " << req_index
            << " is out of bounds [0, " << in_t->size() << "]")
        return nullptr;
    }

    // get the time
    double t = 0.;
    in_t->get(req_index, t);

    // slice axes on the requested extent
    p_teca_variant_array out_x = in_x->new_copy(req_extent[0], req_extent[1]);
    p_teca_variant_array out_y = in_y->new_copy(req_extent[2], req_extent[3]);
    p_teca_variant_array out_z = in_z->new_copy(req_extent[4], req_extent[5]);

    // create output dataset
    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();

    std::string x_variable = this->x_axis_variable.empty() ? "x" : this->x_axis_variable;
    std::string y_variable = this->y_axis_variable.empty() ? "y" : this->y_axis_variable;
    std::string z_variable = this->z_axis_variable.empty() ? "z" : this->z_axis_variable;

    mesh->set_x_coordinates(x_variable, out_x);
    mesh->set_y_coordinates(y_variable, out_y);
    mesh->set_z_coordinates(z_variable, out_z);

    // get the calendar
    std::string calendar;
    std::string units;
    teca_metadata atts;
    this->internals->metadata.get("attributes", atts);
    atts.get("calendar", calendar);
    atts.get("units", units);

    // set metadata
    mesh->set_whole_extent(md_whole_extent);
    mesh->set_extent(req_extent);
    mesh->set_time_step(req_index);
    mesh->set_time(t);
    mesh->set_calendar(calendar);
    mesh->set_time_units(units);

    teca_metadata &mesh_md = mesh->get_metadata();
    mesh_md.set("index_request_key", request_key);

    // generate fields over the requested subset
    std::vector<field_generator_t>::iterator it = this->field_generators.begin();
    std::vector<field_generator_t>::iterator end = this->field_generators.end();
    for (; it != end; ++it)
    {
        p_teca_variant_array f_xyzt = it->generator(out_x, out_y, out_z, t);
        mesh->get_point_arrays()->append(it->name, f_xyzt);
    }

    // pass the attributes
    mesh_md.set("attributes", atts);

    return mesh;
}
