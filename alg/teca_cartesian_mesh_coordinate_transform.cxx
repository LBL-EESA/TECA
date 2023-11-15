#include "teca_cartesian_mesh_coordinate_transform.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG

using namespace teca_variant_array_util;

struct teca_cartesian_mesh_coordinate_transform::internals_t
{
    internals_t() : bounds_in{0.0}, bounds_out{0.0} {}

    void clear();

    /** transform the input such that it covers x_out_0 to x_out_1 while
     * maintaining relative relationships between points
     */
    template <typename coord_t>
    static
    void transform_axis(coord_t *x_out, const coord_t *x_in, size_t nx,
        coord_t x_out_0, coord_t x_out_1);

    /// check that the requested bounds are valid for a given input
    static
    int validate_target_bounds(char ax_name,
        const const_p_teca_variant_array &ax_in, const double *tgt_bounds);

    /** allocate buffers and apply the transform. the transform is skipped
     * if the rquested bounds are invalid (higher bounds is less than lower bound)
     */
    static
    void transform_axes(p_teca_variant_array &ax_out,
        const const_p_teca_variant_array &ax_in, const double *tgt_bounds);


    double bounds_in[6];
    double bounds_out[6];

    teca_metadata coordinates_in;
    teca_metadata coordinates_out;

    std::string x_axis_variable_out;
    std::string y_axis_variable_out;
    std::string z_axis_variable_out;

    teca_metadata x_axis_attributes_out;
    teca_metadata y_axis_attributes_out;
    teca_metadata z_axis_attributes_out;
};

// --------------------------------------------------------------------------
void teca_cartesian_mesh_coordinate_transform::internals_t::clear()
{
    for (int i = 0; i < 6; ++i)
        bounds_in[i] = 0.0;

    for (int i = 0; i < 6; ++i)
        bounds_out[i] = 0.0;

    this->coordinates_in.clear();
    this->coordinates_out.clear();

    this->x_axis_variable_out = "";
    this->y_axis_variable_out = "";
    this->z_axis_variable_out = "";

    this->x_axis_attributes_out.clear();
    this->y_axis_attributes_out.clear();
    this->z_axis_attributes_out.clear();
}

// --------------------------------------------------------------------------
template <typename coord_t>
void teca_cartesian_mesh_coordinate_transform::internals_t::transform_axis(
    coord_t *x_out, const coord_t *x_in, size_t nx, coord_t x_out_0, coord_t x_out_1)
{
    // transform the input onto [0, 1] and then
    // shift and scale to the target range [x_out_0, x_out_1]
    coord_t x_in_min = x_in[0];
    coord_t x_in_max = x_in[nx - 1];
    coord_t delta_x_in = x_in_max - x_in_min;
    coord_t delta_x_out = x_out_1 - x_out_0;
    coord_t scale_fac = nx < 2 ? coord_t(0) : delta_x_out / delta_x_in;

    for (size_t i = 0; i < nx; ++i)
        x_out[i] = (x_in[i] - x_in_min) * scale_fac + x_out_0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_coordinate_transform::internals_t::validate_target_bounds(
    char ax_name, const const_p_teca_variant_array &ax_in, const double *tgt_bounds)
{
    size_t ax_size = ax_in->size();
    bool tgt_bounds_valid = !(tgt_bounds[1] < tgt_bounds[0]);

    // check that the bounds specified in each axis direction are different
    // skip this check when the bounds are invlaid, or the input axis has
    // only 1 value
    if (tgt_bounds_valid && (ax_size > 1) &&
        teca_coordinate_util::equal(tgt_bounds[0], tgt_bounds[1]))
    {
        TECA_ERROR("Invlaid " << ax_name << "-axis target bounds specified ["
            << tgt_bounds[0] << ", " << tgt_bounds[1]
            << "] the values are equal but there are " << ax_size << " values.")

        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_coordinate_transform::internals_t::transform_axes(
    p_teca_variant_array &ax_out, const const_p_teca_variant_array &ax_in,
    const double *tgt_bounds)
{
        if(!(tgt_bounds[1] < tgt_bounds[0]))
        {
            // allocate output
            size_t ax_size = ax_in->size();
            ax_out = ax_in->new_instance(ax_size);

            VARIANT_ARRAY_DISPATCH(ax_out.get(),

                auto [sp_ax_in, p_ax_in] = get_host_accessible<CTT>(ax_in);
                auto [p_ax_out] = data<TT>(ax_out);

                sync_host_access_any(ax_in);

                // transform the axis
                teca_cartesian_mesh_coordinate_transform::internals_t::transform_axis(
                    p_ax_out, p_ax_in, ax_size, NT(tgt_bounds[0]), NT(tgt_bounds[1]));
            )
        }
        else
        {
            // the requested bounds are invalid, pass the current axis through
            ax_out = ax_in->new_copy();
        }
}


// --------------------------------------------------------------------------
teca_cartesian_mesh_coordinate_transform::teca_cartesian_mesh_coordinate_transform()
    : target_bounds({1.0,0.0, 1.0,0.0, 1.0,0.0})
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->internals = new teca_cartesian_mesh_coordinate_transform::internals_t;
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_coordinate_transform::~teca_cartesian_mesh_coordinate_transform()
{
    delete this->internals;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_coordinate_transform::set_modified()
{
    this->internals->clear();
    this->teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh_coordinate_transform::set_target_bounds(const teca_metadata &md)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
    {
        TECA_ERROR("Failed to set target_bounds metadata is missing"
            " coordinates key")
        return -1;
    }

    // set target bounds to pass through
    this->target_bounds[0] = 1;
    this->target_bounds[1] = 0;
    this->target_bounds[2] = 1;
    this->target_bounds[3] = 0;
    this->target_bounds[4] = 1;
    this->target_bounds[5] = 0;

    // set the bounds from the coordinate axes that are found
    const_p_teca_variant_array x = coords.get("x");
    if (x)
    {
        unsigned long nx = x->size();
        x->get(0, this->target_bounds[0]);
        x->get(nx - 1, this->target_bounds[1]);
    }

    const_p_teca_variant_array y = coords.get("y");
    if (y)
    {
        unsigned long ny = y->size();
        y->get(0, this->target_bounds[2]);
        y->get(ny - 1, this->target_bounds[3]);
    }

    const_p_teca_variant_array z = coords.get("z");
    if (z)
    {
        unsigned long nz = z->size();
        z->get(0, this->target_bounds[4]);
        z->get(nz - 1, this->target_bounds[5]);
    }

    if (!x && !y && !z)
    {
        TECA_ERROR("failed to set target_bounds cooridinate metadata"
            " is missing x, y, and z coordinate arrays")
        return -1;
    }

    return 0;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cartesian_mesh_coordinate_transform::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_coordinate_transform":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, target_bounds,
            "6 double precision values that define the output coordinate axis"
            " bounds, specified in the following order : [x0 x1 y0 y1 z0 z1]."
            " The Cartesian mesh is transformed such that its coordinatres span"
            " the specified target bounds while maintaining relative spacing of"
            " original input coordinate points. Pass [1, 0] for each axis that"
            " should not be transformed.")
        TECA_POPTS_GET(std::string, prefix, x_axis_variable,
            "Set the name of variable that has x axis coordinates. If not"
            " provided, the name passed through unchanged.")
        TECA_POPTS_GET(std::string, prefix, y_axis_variable,
            "Set the name of variable that has y axis coordinates. If not"
            " provided, the name is passed through unchanged.")
        TECA_POPTS_GET(std::string, prefix, z_axis_variable,
            "Set the name of variable that has z axis coordinates. If not"
            " provided, the name is passed through unchanged.")
        TECA_POPTS_GET(std::string, prefix, x_axis_units,
            "Set the units of the x-axis coordinates. If not provided the"
            " units are passed through unchanged.")
        TECA_POPTS_GET(std::string, prefix, y_axis_units,
            "Set the units of the y-axis coordinates. If not provided the"
            " units are passed through unchanged.")
        TECA_POPTS_GET(std::string, prefix, z_axis_units,
            "Set the units of the z-axis coordinates. If not provided the"
            " units are passed through unchanged.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_coordinate_transform::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<double>, prefix, target_bounds)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_variable)
    TECA_POPTS_SET(opts, std::string, prefix, x_axis_units)
    TECA_POPTS_SET(opts, std::string, prefix, y_axis_units)
    TECA_POPTS_SET(opts, std::string, prefix, z_axis_units)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_coordinate_transform::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cartesian_mesh_coordinate_transform::get_output_metadata"
        << std::endl;
#endif
    (void)port;

    if (this->internals->coordinates_out)
    {
        // update coordinates and bounds
        teca_metadata md_out = input_md[0];
        md_out.set("coordinates", this->internals->coordinates_out);
        md_out.set("bounds", this->internals->bounds_out);

        // update the attributes
        teca_metadata atts_out;
        md_out.get("attributes", atts_out);

        atts_out.set(this->internals->x_axis_variable_out, this->internals->x_axis_attributes_out);
        atts_out.set(this->internals->y_axis_variable_out, this->internals->y_axis_attributes_out);
        atts_out.set(this->internals->z_axis_variable_out, this->internals->z_axis_attributes_out);

        md_out.set("attributes", atts_out);

        return md_out;
    }

    // check that bounds for each cooridnate axes are specified
    if (this->target_bounds.size() != 6)
    {
        TECA_FATAL_ERROR("Invalid target_bounds. " << this->target_bounds.size()
            << "  were specified while 6 are needed.")
        return teca_metadata();
    }

    // copy the metadata
    teca_metadata out_md(input_md[0]);

    // get the input coordinate axes
    const_p_teca_variant_array x_in;
    const_p_teca_variant_array y_in;
    const_p_teca_variant_array z_in;

    if (out_md.get("coordinates", this->internals->coordinates_in)
        || !(x_in = this->internals->coordinates_in.get("x"))
        || !(y_in = this->internals->coordinates_in.get("y"))
        || !(z_in = this->internals->coordinates_in.get("z")))
    {
        TECA_FATAL_ERROR("The input metadata has invalid coordinates")
        this->internals->clear();
        return teca_metadata();
    }

    // get the input bounds
    teca_coordinate_util::get_cartesian_mesh_bounds(x_in,
        y_in, z_in, this->internals->bounds_in);

    const double *tgt_bounds = this->target_bounds.data();

    // check the bounds request
    if (internals_t::validate_target_bounds('x', x_in, tgt_bounds    )
        || internals_t::validate_target_bounds('y', y_in, tgt_bounds + 2)
        || internals_t::validate_target_bounds('z', z_in, tgt_bounds + 4))
    {
        TECA_FATAL_ERROR("Invalid bounds requested")
        this->internals->clear();
        return teca_metadata();
    }

    // transform the coordinate axes
    p_teca_variant_array x_out;
    p_teca_variant_array y_out;
    p_teca_variant_array z_out;

    internals_t::transform_axes(x_out, x_in, tgt_bounds    );
    internals_t::transform_axes(y_out, y_in, tgt_bounds + 2);
    internals_t::transform_axes(z_out, z_in, tgt_bounds + 4);

    // set the new axes into the coordinates object
    this->internals->coordinates_out = this->internals->coordinates_in;

    this->internals->coordinates_out.set("x", x_out);
    this->internals->coordinates_out.set("y", y_out);
    this->internals->coordinates_out.set("z", z_out);

    // update the bounds in the output metadata
    teca_coordinate_util::get_cartesian_mesh_bounds(x_out,
        y_out, z_out, this->internals->bounds_out);

    out_md.set("bounds", this->internals->bounds_out, 6);

    // get the input coordinate axis names
    std::string x_axis_variable_in;
    std::string y_axis_variable_in;
    std::string z_axis_variable_in;
    if (this->internals->coordinates_in.get("x_variable", x_axis_variable_in)
        || this->internals->coordinates_in.get("y_variable", y_axis_variable_in)
        || this->internals->coordinates_in.get("z_variable", z_axis_variable_in))
    {
        TECA_FATAL_ERROR("Failed to get the coordinate axis variables")
        this->internals->clear();
        return teca_metadata();
    }

    // update the output axis variable names
    this->internals->x_axis_variable_out =
        this->x_axis_variable.empty() ? x_axis_variable_in : this->x_axis_variable;

    this->internals->y_axis_variable_out =
        this->y_axis_variable.empty() ? y_axis_variable_in : this->y_axis_variable;

    this->internals->z_axis_variable_out =
        this->z_axis_variable.empty() ? z_axis_variable_in : this->z_axis_variable;

    this->internals->coordinates_out.set("x_variable",
        this->internals->x_axis_variable_out);

    this->internals->coordinates_out.set("y_variable",
        this->internals->y_axis_variable_out);
    this->internals->coordinates_out.set("z_variable",
        this->internals->z_axis_variable_out);

    // pass the updated coordinate system
    out_md.set("coordinates", this->internals->coordinates_out);

    // get the input attributes
    teca_metadata atts;
    if (out_md.get("attributes", atts))
    {
        TECA_FATAL_ERROR("Failed to get the coordinate variables attributes")
        this->internals->clear();
        return teca_metadata();
    }

    // get coordinates attributes. some of these might be empty. that is OK.
    atts.get(x_axis_variable_in, this->internals->x_axis_attributes_out);
    atts.get(y_axis_variable_in, this->internals->y_axis_attributes_out);
    atts.get(z_axis_variable_in, this->internals->z_axis_attributes_out);

    // x axis attributes
    // update the description to note the transform
    if (!(tgt_bounds[1] < tgt_bounds[0]))
    {
        std::ostringstream oss;
        oss << x_axis_variable_in << " transformed from ["
            << this->internals->bounds_in[0] << ", " << this->internals->bounds_in[1]
            << "] to [" << this->internals->bounds_out[0] << ", "
            << this->internals->bounds_out[1] << "]";

        this->internals->x_axis_attributes_out.set("description", oss.str());
    }

    // update the units
    if (!this->x_axis_units.empty())
        this->internals->x_axis_attributes_out.set("units", this->x_axis_units);

    // remove the long name
    if (!this->x_axis_variable.empty())
        this->internals->x_axis_attributes_out.remove("long_name");

    // y axis attributes
    // update the description to note the transform
    if (!(tgt_bounds[3] < tgt_bounds[2]))
    {
        std::ostringstream oss;
        oss << y_axis_variable_in << " transformed from ["
            << this->internals->bounds_in[2] << ", " << this->internals->bounds_in[3]
            << "] to [" << this->internals->bounds_out[2] << ", "
            << this->internals->bounds_out[3] << "]";

        this->internals->y_axis_attributes_out.set("description", oss.str());
    }

    // update the units
    if (!this->y_axis_units.empty())
        this->internals->y_axis_attributes_out.set("units", this->y_axis_units);

    // remove the long name
    if (!this->y_axis_variable.empty())
        this->internals->y_axis_attributes_out.remove("long_name");

    // z axis attributes
    // update the description to note the transform
    if (!(tgt_bounds[5] < tgt_bounds[4]))
    {
        std::ostringstream oss;
        oss << z_axis_variable_in << " transformed from ["
            << this->internals->bounds_in[4] << ", " << this->internals->bounds_in[5]
            << "] to [" << this->internals->bounds_out[4] << ", "
            << this->internals->bounds_out[5] << "]";

        this->internals->z_axis_attributes_out.set(
            "description", oss.str());
    }

    // update the units
    if (!this->z_axis_units.empty())
        this->internals->z_axis_attributes_out.set("units", this->z_axis_units);

    // remove the long name
    if (!this->z_axis_variable.empty())
        this->internals->z_axis_attributes_out.remove("long_name");

    // set the output attributes
    atts.set(this->internals->x_axis_variable_out, this->internals->x_axis_attributes_out);
    atts.set(this->internals->y_axis_variable_out, this->internals->y_axis_attributes_out);
    atts.set(this->internals->z_axis_variable_out, this->internals->z_axis_attributes_out);

    // pass the updated attributes
    out_md.set("attributes", atts);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_cartesian_mesh_coordinate_transform::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    if (request.has("bounds"))
    {
        // get the requested bounds. these are specified in the output
        // coordinate system
        double bounds[6] = {0.0};
        request.get("bounds", bounds);

        // get the input coordinate axes
        const_p_teca_variant_array x_in;
        const_p_teca_variant_array y_in;
        const_p_teca_variant_array z_in;

        if (!(x_in = this->internals->coordinates_in.get("x"))
            || !(y_in = this->internals->coordinates_in.get("y"))
            || !(z_in = this->internals->coordinates_in.get("z")))
        {
            TECA_FATAL_ERROR("The input metadata has invalid coordinates")
            return up_reqs;
        }

        // get the transformed coordinate axes
        p_teca_variant_array x_out;
        p_teca_variant_array y_out;
        p_teca_variant_array z_out;

        if (!(x_out = this->internals->coordinates_out.get("x"))
            || !(y_out = this->internals->coordinates_out.get("y"))
            || !(z_out = this->internals->coordinates_out.get("z")))
        {
            TECA_FATAL_ERROR("The input metadata has invalid coordinates")
            return up_reqs;
        }

        // get the indices of bounding box. these can be used to invert the
        // transform.
        unsigned long extent[6] = {0ul};
        if (teca_coordinate_util::bounds_to_extent(bounds, x_out, y_out, z_out, extent))
        {
            TECA_FATAL_ERROR("The requested bounds " << bounds[0] << ", " << bounds[1]
                << ", " << bounds[2] << ", "  << bounds[3] << ", " << bounds[4]
                << ", " << bounds[5] << "] were not found in the transformed coordinates")
            return up_reqs;
        }

        // using the indices look up the bounding box in the input coordinate system
        double bounds_up[6] = {0.0};
        x_in->get(extent[0], bounds_up[0]);
        x_in->get(extent[1], bounds_up[1]);
        y_in->get(extent[2], bounds_up[2]);
        y_in->get(extent[3], bounds_up[3]);
        z_in->get(extent[4], bounds_up[4]);
        z_in->get(extent[5], bounds_up[5]);

        teca_metadata req(request);
        req.set("bounds", bounds_up, 6);

        up_reqs.push_back(req);
    }
    else
    {
        up_reqs.push_back(request);
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_coordinate_transform::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_cartesian_mesh_coordinate_transform::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    p_teca_cartesian_mesh in_target
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(input_data[0]));

    if (!in_target)
    {
        TECA_FATAL_ERROR("invalid input dataset")
        return nullptr;
    }

    // get the transformed coordinate axes
    p_teca_variant_array x_out;
    p_teca_variant_array y_out;
    p_teca_variant_array z_out;

    if (!(x_out = this->internals->coordinates_out.get("x"))
        || !(y_out = this->internals->coordinates_out.get("y"))
        || !(z_out = this->internals->coordinates_out.get("z")))
    {
        TECA_FATAL_ERROR("The cached metadata has invalid coordinates")
        return nullptr;
    }


    // pass input through via shallow copy
    p_teca_cartesian_mesh target = teca_cartesian_mesh::New();
    target->shallow_copy(in_target);

    // get the subset
    unsigned long extent[6] = {0ul};
    in_target->get_extent(extent);

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    // get the input cooridnates
    p_teca_variant_array x_in = target->get_x_coordinates();
    p_teca_variant_array y_in = target->get_y_coordinates();
    p_teca_variant_array z_in = target->get_z_coordinates();

    // allocate the subset output coordinates
    p_teca_variant_array x_out_sub = x_in->new_instance(nx);
    p_teca_variant_array y_out_sub = y_in->new_instance(ny);
    p_teca_variant_array z_out_sub = z_in->new_instance(nz);

    // copy the subset
    VARIANT_ARRAY_DISPATCH(x_out.get(),

        auto [p_xo, p_yo, p_zo] = data<TT>(x_out, y_out, z_out);
        auto [p_xos, p_yos, p_zos] = data<TT>(x_out_sub, y_out_sub, z_out_sub);

        for (unsigned long i = 0; i < nx; ++i)
            p_xos[i] = p_xo[extent[0] + i];

        for (unsigned long i = 0; i < ny; ++i)
            p_yos[i] = p_yo[extent[2] + i];

        for (unsigned long i = 0; i < nz; ++i)
            p_zos[i] = p_zo[extent[4] + i];
        )

    // update the mesh
    target->set_x_coordinates(this->internals->x_axis_variable_out, x_out_sub);
    target->set_y_coordinates(this->internals->y_axis_variable_out, y_out_sub);
    target->set_z_coordinates(this->internals->z_axis_variable_out, z_out_sub);

    teca_metadata atts_out;
    target->get_attributes(atts_out);

    atts_out.set(this->internals->x_axis_variable_out, this->internals->x_axis_attributes_out);
    atts_out.set(this->internals->y_axis_variable_out, this->internals->y_axis_attributes_out);
    atts_out.set(this->internals->z_axis_variable_out, this->internals->z_axis_attributes_out);

    target->set_attributes(atts_out);

    return target;
}
