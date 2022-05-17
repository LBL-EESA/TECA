#include "teca_gradient.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

// #define TECA_DEBUG

namespace {

template <typename num_t>
constexpr num_t deg_to_rad() { return num_t(M_PI)/num_t(180); }

template <typename num_t>
constexpr num_t earth_radius() { return num_t(6371.0e3); }

// compute the gradient using second-order, centered finite difference
// assumes fixed mesh spacing. here we add periodic bc in lon
// and apply unit stride vector optimization strategy to loops
template <typename num_t, typename pt_t>
void gradient(num_t *grad_x, num_t *grad_y, const pt_t *lon, const pt_t *lat,
    const num_t *f, unsigned long n_lon,
    unsigned long n_lat, bool periodic_lon=true)
{
    // initialize an array to hold the latitude-dependent dx values
    size_t n_bytes = n_lat*sizeof(num_t);
    num_t *delta_x = static_cast<num_t*>(malloc(n_bytes));

    // delta lon as a function of latitude
    num_t delta_x_equator = num_t(2)*
        (lon[1] - lon[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();
    for (unsigned long j = 0; j < n_lat; ++j)
        delta_x[j] = delta_x_equator * cos(lat[j] * deg_to_rad<num_t>());

    // delta lat
    num_t dy = num_t(2)*
        (lat[1] - lat[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();

    unsigned long max_i = n_lon - 1;
    unsigned long max_j = n_lat - 1;

    // loop over all latitudes
    // (boundary conditions for grad_y are taken care of
    //  at the end)
    for (unsigned long j = 0; j <= max_j; ++j)
    {
        unsigned long jj = j*n_lon;
        const num_t *f_y2 = f + jj + n_lon;
        const num_t *f_y0 = f + jj - n_lon;
        const num_t *f_x2 = f + jj + 1;
        const num_t *f_x0 = f + jj - 1;
        num_t *gx = grad_x + jj;
        num_t *gy = grad_y + jj;
        num_t dx = delta_x[j];

        // loop over all longitudes
        for (unsigned long i = 1; i < max_i; ++i)
        {
            // calculate the x-gradient
            gx[i] = (f_x2[i] - f_x0[i]) / dx;

            // calculate the y-gradient, only if
            // we aren't at the poles
            if ( (j != 0) && (j != max_j))
                gy[i] = (f_y2[i] - f_y0[i]) / dy;
        }
    }

    if (periodic_lon)
    {
        // loop over all latitudes for the western boundary
        // (boundary conditions for grad_y are taken care of
        //  at the end)
        for (unsigned long j = 0; j <= max_j; ++j)
        {
            unsigned long jj = j*n_lon;
            const num_t *f_y2 = f + jj + n_lon;
            const num_t *f_y0 = f + jj - n_lon;
            // set values so that df/dx accounts for
            // wrap-around
            const num_t *f_x2 = f + jj + 1;
            const num_t *f_x0 = f + jj + max_i;
            num_t *gx = grad_x + jj;
            num_t *gy = grad_y + jj;
            num_t dx = delta_x[j];

            // calculate the x-gradient
            gx[0] = (f_x2[0] - f_x0[0]) / dx;

            // calculate the y-gradient, only if
            // we aren't at the poles
            if ( (j != 0) && (j != max_j))
                gy[0] = (f_y2[0] - f_y0[0]) / dy;
        }

        // loop over all latitudes for the eastern boundary
        // (boundary conditions for grad_y are taken care of
        //  at the end)
        for (unsigned long j = 0; j <= max_j; ++j)
        {
            unsigned long jj = j*n_lon;
            const num_t *f_y2 = f + jj + max_i + n_lon;
            const num_t *f_y0 = f + jj + max_i - n_lon;
            // set values so that df/dx accounts for
            // wrap-around
            const num_t *f_x2 = f + jj;
            const num_t *f_x0 = f + jj + max_i - 1;
            num_t *gx = grad_x + jj + max_i;
            num_t *gy = grad_y + jj + max_i;
            num_t dx = delta_x[j];

            // calculate the x-gradient
            gx[0] = (f_x2[0] - f_x0[0]) / dx;

            // calculate the y-gradient, only if
            // we aren't at the poles
            if ( (j != 0) && (j != max_j))
                gy[0] = (f_y2[0] - f_y0[0]) / dy;
        }
    }
    else
    {
        // zero it out
        for (unsigned long j = 1; j < max_j; ++j)
        {
            grad_x[j*n_lon] = num_t();
            grad_y[j*n_lon] = num_t();
        }

        for (unsigned long j = 1; j < max_j; ++j)
        {
            grad_x[j*n_lon + max_i] = num_t();
            grad_y[j*n_lon + max_i] = num_t();
        }
    }

    // extend y-derivative values onto the south pole
    num_t *dest = grad_y;
    num_t *src = grad_y + n_lon;
    for (unsigned long i = 0; i < n_lon; ++i)
        dest[i] = src[i];

    // extend y-derivative values onto the north pole
    dest = grad_y + max_j*n_lon;
    src = dest - n_lon;
    for (unsigned long i = 0; i < n_lon; ++i)
        dest[i] = src[i];

    free(delta_x);

    return;
}
};


// --------------------------------------------------------------------------
teca_gradient::teca_gradient() :
    scalar_field(),
    gradient_field_x("gradient_x"),
    gradient_field_y("gradient_y")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_gradient::~teca_gradient()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_gradient::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_gradient":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, scalar_field,
            "array containing the scalar field")
        TECA_POPTS_GET(std::string, prefix, gradient_field_x,
            "array to store the x-direction of the computed gradient")
        TECA_POPTS_GET(std::string, prefix, gradient_field_y,
            "array to store the y-direction of the computed gradient")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_gradient::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, scalar_field)
    TECA_POPTS_SET(opts, std::string, prefix, gradient_field_x)
    TECA_POPTS_SET(opts, std::string, prefix, gradient_field_y)
}
#endif

// --------------------------------------------------------------------------
std::string teca_gradient::get_scalar_field(
    const teca_metadata &request)
{
    std::string scalar_var = this->scalar_field;

    if (scalar_var.empty() &&
        request.has("teca_gradient::scalar_field"))
            request.get("teca_gradient::scalar_field", scalar_var);

    return scalar_var;
}

// --------------------------------------------------------------------------
std::string teca_gradient::get_gradient_field_x(
    const teca_metadata &request)
{
    std::string grad_x_var = this->gradient_field_x;

    if (grad_x_var.empty() &&
        request.has("teca_gradient::gradient_field_x"))
            request.get("teca_gradient::gradient_field_x", grad_x_var);
    else if (grad_x_var.empty() &&
        !request.has("teca_gradient::gradient_field_x"))
        // set a default
        grad_x_var = this->scalar_field + "_gradient_x";

    return grad_x_var;
}

// --------------------------------------------------------------------------
std::string teca_gradient::get_gradient_field_y(
    const teca_metadata &request)
{
    std::string grad_y_var = this->gradient_field_y;

    if (grad_y_var.empty() &&
        request.has("teca_gradient::gradient_field_y"))
            request.get("teca_gradient::gradient_field_y", grad_y_var);
    else if (grad_y_var.empty() &&
        !request.has("teca_gradient::gradient_field_y"))
        // set a default
        grad_y_var = this->scalar_field + "_gradient_y";

    return grad_y_var;
}


// --------------------------------------------------------------------------
teca_metadata teca_gradient::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_gradient::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the arrays we will generate
    teca_metadata out_md(input_md[0]);
    out_md.append("variables", this->get_gradient_field_x(out_md));
    out_md.append("variables", this->get_gradient_field_y(out_md));

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    teca_metadata scalar_field_atts;
    if (attributes.get(this->get_scalar_field(out_md), scalar_field_atts))
    {
        TECA_WARNING("Failed to get scalar field \"" << 
            this->get_scalar_field(out_md) << "\" attributes. "
            "Writing the result will not be possible")
    }
    else
    {
        // copy the attributes from the input. this will capture the
        // data type, size, units, etc.
        teca_array_attributes grad_x_atts(scalar_field_atts);

        // update units, long_name, and description.
        grad_x_atts.units += " m-1";
        grad_x_atts.long_name += " gradient";

        grad_x_atts.description =
            std::string("The x-component of the gradient of '");
        grad_x_atts.description += this->get_scalar_field(out_md);
        grad_x_atts.description += "'";

        attributes.set(this->get_gradient_field_x(out_md),
            (teca_metadata)grad_x_atts);

        // copy the attributes from the input. this will capture the
        // data type, size, units, etc.
        teca_array_attributes grad_y_atts(scalar_field_atts);

        // update units, long_name, and description.
        grad_y_atts.units += " m-1";
        grad_y_atts.long_name += " gradient";

        grad_y_atts.description =
            std::string("The y-component of the gradient of '");
        grad_y_atts.description += this->get_scalar_field(out_md);
        grad_y_atts.description += "'";

        attributes.set(this->get_gradient_field_y(out_md),
            (teca_metadata)grad_y_atts);


        out_md.set("attributes", attributes);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_gradient::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the arrays we need to request
    std::string scalar_var = this->get_scalar_field(request);
    if (scalar_var.empty())
    {
        TECA_FATAL_ERROR("scalar field array was not specified")
        return up_reqs;
    }

    std::string grad_x_var = this->get_gradient_field_x(request);
    if (grad_x_var.empty())
    {
        TECA_FATAL_ERROR("gradient x array name was not specified")
        return up_reqs;
    }

    std::string grad_y_var = this->get_gradient_field_y(request);
    if (grad_y_var.empty())
    {
        TECA_FATAL_ERROR("gradient y array name was not specified")
        return up_reqs;
    }


    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->scalar_field);

    // capture the array we produce
    arrays.erase(this->get_gradient_field_x(request));
    arrays.erase(this->get_gradient_field_y(request));

    // update the request
    req.set("arrays", arrays);

    // send it up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_gradient::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_gradient::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("teca_cartesian_mesh is required")
        return nullptr;
    }

    // get the scalar field array
    std::string scalar_var = this->get_scalar_field(request);

    if (scalar_var.empty())
    {
        TECA_FATAL_ERROR("scalar_field was not specified")
        return nullptr;
    }

    const_p_teca_variant_array scalar
        = in_mesh->get_point_arrays()->get(scalar_var);

    if (!scalar)
    {
        TECA_FATAL_ERROR("requested array \"" << scalar_var 
            << "\" not present.")
        return nullptr;
    }

    // get the input coordinate arrays
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();

    if (!lon || !lat)
    {
        TECA_FATAL_ERROR("lat lon mesh coordinates not present.")
        return nullptr;
    }

    // allocate the output arrays
    p_teca_variant_array grad_x = scalar->new_instance();
    p_teca_variant_array grad_y = scalar->new_instance();
    grad_x->resize(scalar->size());
    grad_y->resize(scalar->size());

    // compute gradient
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        lon.get(), 1,

        auto sp_lon = dynamic_cast<TT1*>(lon.get())->get_cpu_accessible();
        const NT1 *p_lon = sp_lon.get();

        auto sp_lat = dynamic_cast<TT1*>(lat.get())->get_cpu_accessible();
        const NT1 *p_lat = sp_lat.get();

        NESTED_TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            grad_x.get(), 2,

            auto sp_scalar = 
                dynamic_cast<const TT2*>(scalar.get())->get_cpu_accessible();
            const NT2 *p_scalar = sp_scalar.get();

            auto sp_grad_x = 
                dynamic_cast<TT2*>(grad_x.get())->get_cpu_accessible();
            NT2 *p_grad_x = sp_grad_x.get();

            auto sp_grad_y = 
                dynamic_cast<TT2*>(grad_y.get())->get_cpu_accessible();
            NT2 *p_grad_y = sp_grad_y.get();


            ::gradient(p_grad_x, p_grad_y, p_lon, p_lat,
                p_scalar, lon->size(), lat->size());
            )
        )

    // create the output mesh, pass everything through, and
    // add the gradient array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    out_mesh->get_point_arrays()->append(
        this->get_gradient_field_x(request), grad_x);

    out_mesh->get_point_arrays()->append(
        this->get_gradient_field_y(request), grad_y);

    return out_mesh;
}
