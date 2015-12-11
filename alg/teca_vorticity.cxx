#include "teca_vorticity.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

namespace {

template <typename num_t>
constexpr num_t deg_to_rad() { return num_t(M_PI)/num_t(180); }

template <typename num_t>
constexpr num_t earth_radius() { return num_t(6371.0e3); }

// compute vorticicty
template <typename num_t, typename pt_t>
void vorticity(num_t *w, const pt_t *lat, const pt_t *lon,
    const num_t *vx, const num_t *vy, unsigned long nx, unsigned long ny)
{
    // compute dx from degrees longitude
    num_t *dx = static_cast<num_t*>(malloc(nx*sizeof(num_t)));
    num_t dlon = (lon[1]- lon[0])*deg_to_rad<num_t>();
    for (unsigned long i = 0; i < nx; ++i)
        dx[i] = earth_radius<num_t>() * cos(lat[i]*deg_to_rad<num_t>()) * dlon;

    // compute dy from degrees latitude
    unsigned long max_j = ny - 1;
    num_t *dy = static_cast<num_t*>(malloc(ny*sizeof(num_t)));
    for (unsigned long i = 1; i < max_j; ++i)
        dy[i] = num_t(0.5)*earth_radius<num_t>()*deg_to_rad<num_t>()
            *(lat[i-1] - lat[i+1]);
    dy[0] = dy[1];
    dy[max_j] = dy[max_j - 1];

    // compute vorticity
    unsigned long nxy = nx*ny;
    memset(w, 0, nxy*sizeof(num_t));
    unsigned long max_i = nx - 1;
    for (unsigned long j = 1; j < max_j; ++j)
    {
        // TODO -- rewrite this in terms of unit stride passes
        // so that the compiler will auto-vectorize it
        unsigned long jj = j*nx;
        unsigned long jj0 = jj - nx;
        unsigned long jj1 = jj + nx;
        for (unsigned long i = 1; i < max_i; ++i)
        {
            w[jj+i] = num_t(0.5)*((vy[jj+i+1] - vy[jj+i-1])/dx[i]
                - (vx[jj0+i] - vx[jj1+i])/dy[j]);
        }
    }

    free(dx);
    free(dy);

    return;
}
};


// --------------------------------------------------------------------------
teca_vorticity::teca_vorticity() :
    component_0_variable(), component_1_variable(),
    vorticity_variable("vorticity")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_vorticity::~teca_vorticity()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vorticity::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for " + prefix + "(teca_vorticity)");

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable, "array containg x-component of the vector")
        TECA_POPTS_GET(std::string, prefix, component_1_variable, "array containg y-component of the vector")
        TECA_POPTS_GET(std::string, prefix, vorticity_variable, "array to store the computed vorticity in")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vorticity::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_1_variable)
    TECA_POPTS_SET(opts, std::string, prefix, vorticity_variable)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_vorticity::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_vorticity::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);
    out_md.append("variables", this->vorticity_variable);

    return out_md;
}
// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_vorticity::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void) port;

    vector<teca_metadata> up_reqs;

    // do some error checking.
    if (this->component_0_variable.empty() || this->component_1_variable.empty())
    {
        TECA_ERROR("the vector component arrays were not specified")
        return up_reqs;
    }

    // we could also check the input_md for these arrays...
    // but if not the upstream will handle it.

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);
    std::vector<std::string> arrays;
    req.get("arrays", arrays);
    arrays.push_back(this->component_0_variable);
    arrays.push_back(this->component_1_variable);
    req.insert("arrays", arrays);

    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vorticity::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_vorticity::execute" << endl;
#endif
    (void)port;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("teca_cartesian_mesh is required")
        return nullptr;
    }

    // get the input vector component arrays
    const_p_teca_variant_array vx
        = in_mesh->get_point_arrays()->get(this->component_0_variable);

    if (!vx)
    {
        TECA_ERROR("x-component array \""
            << this->component_0_variable << "\" not found.")
        return nullptr;
    }

    const_p_teca_variant_array vy
        = in_mesh->get_point_arrays()->get(this->component_1_variable);

    if (!vy)
    {
        TECA_ERROR("y-component array \""
            << this->component_1_variable << "\" not found.")
        return nullptr;
    }

    // get the inpute coordinate arrays
    const_p_teca_variant_array x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array y = in_mesh->get_y_coordinates();

    if (!x || !y)
    {
        TECA_ERROR("mesh cooridinates not found.")
        return nullptr;
    }

    unsigned long nx = x->size();
    unsigned long ny = y->size();

    // allocate the output array
    p_teca_variant_array w = vx->new_instance();
    w->resize(vx->size());

    // compute vorticity
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        x.get(), 1,

        const NT1 *px = dynamic_cast<const TT1*>(x.get())->get();
        const NT1 *py = dynamic_cast<const TT1*>(y.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            w.get(), 2,

            const NT2 *pvx = dynamic_cast<const TT2*>(vx.get())->get();
            const NT2 *pvy = dynamic_cast<const TT2*>(vy.get())->get();
            NT2 *pw = dynamic_cast<TT2*>(w.get())->get();

            vorticity(pw, py, px, pvx, pvy, nx, ny);
            )
        )

    // create the output mesh, pass everything through, and
    // add the vorticity array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));
    out_mesh->get_point_arrays()->append(this->vorticity_variable, w);

    return out_mesh;
}
