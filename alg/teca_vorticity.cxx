#include "teca_vorticity.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

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

//#define TECA_DEBUG

namespace {

template <typename num_t>
constexpr num_t deg_to_rad() { return num_t(M_PI)/num_t(180); }

template <typename num_t>
constexpr num_t earth_radius() { return num_t(6371.0e3); }

// compute vorticicty
// this is based on TECA 1's calculation, and hence
// assumes fixed mesh spacing. here we add periodic bc in lon
// and apply unit stride vector optimization strategy to loops
template <typename num_t, typename pt_t>
void vorticity(num_t *w, const pt_t *lon, const pt_t *lat,
    const num_t *u, const num_t *v, unsigned long n_lon,
    unsigned long n_lat, bool periodic_lon=true)
{
    size_t n_bytes = n_lat*sizeof(num_t);
    num_t *delta_u = static_cast<num_t*>(malloc(n_bytes));

    // delta lon as a function of latitude
    num_t d_lon = (lon[1] - lon[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();
    for (unsigned long j = 0; j < n_lat; ++j)
        delta_u[j] = d_lon * cos(lat[j] * deg_to_rad<num_t>());

    // delta lat
    num_t delta_v = (lat[1] - lat[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();
    num_t dv = num_t(2)*delta_v;

    unsigned long max_i = n_lon - 1;
    unsigned long max_j = n_lat - 1;

    // vorticity
    for (unsigned long j = 1; j < max_j; ++j)
    {
        unsigned long jj = j*n_lon;
        const num_t *uu_2 = u + jj + n_lon;
        const num_t *uu_0 = u + jj - n_lon;
        const num_t *vv_2 = v + jj + 1;
        const num_t *vv_0 = v + jj - 1;
        num_t *ww = w + jj;
        num_t du = num_t(2)*delta_u[j];

        for (unsigned long i = 1; i < max_i; ++i)
        {
            ww[i] = (vv_2[i] - vv_0[i]) / du -
                    (uu_2[i] - uu_0[i]) / dv ;
        }
    }

    if (periodic_lon)
    {
        // periodic in longitude
        for (unsigned long j = 1; j < max_j; ++j)
        {
            unsigned long jj = j*n_lon;
            const num_t *uu_2 = u + jj + n_lon;
            const num_t *uu_0 = u + jj - n_lon;
            const num_t *vv_2 = v + jj + 1;
            const num_t *vv_0 = v + jj + max_i;
            num_t *ww = w + jj;
            num_t du = num_t(2)*delta_u[j];

            ww[0] = (vv_2[0] - vv_0[0]) / du -
                    (uu_2[0] - uu_0[0]) / dv ;
        }

        for (unsigned long j = 1; j < max_j; ++j)
        {
            unsigned long jj = j*n_lon;
            const num_t *uu_2 = u + jj + max_i + n_lon;
            const num_t *uu_0 = u + jj + max_i - n_lon;
            const num_t *vv_2 = v + jj;
            const num_t *vv_0 = v + jj + max_i - 1;
            num_t *ww = w + jj + max_i;
            num_t du = num_t(2)*delta_u[j];

            ww[0] = (vv_2[0] - vv_0[0]) / du -
                    (uu_2[0] - uu_0[0]) / dv ;
        }
    }
    else
    {
        // zero it out
        for (unsigned long j = 1; j < max_j; ++j)
            w[j*n_lon] = num_t();

        for (unsigned long j = 1; j < max_j; ++j)
            w[j*n_lon + max_i] = num_t();
    }

    // extend values into lat boundaries
    num_t *dest = w;
    num_t *src = w + n_lon;
    for (unsigned long i = 0; i < n_lon; ++i)
        dest[i] = src[i+n_lon];

    dest = w + max_j*n_lon;
    src = dest - n_lon;
    for (unsigned long i = 0; i < n_lon; ++i)
        dest[i] = src[i];

    free(delta_u);

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
    options_description opts("Options for "
        + (prefix.empty()?"teca_vorticity":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable,
            "array containg lon component of the vector")
        TECA_POPTS_GET(std::string, prefix, component_1_variable,
            "array containg lat component of the vector")
        TECA_POPTS_GET(std::string, prefix, vorticity_variable,
            "array to store the computed vorticity in")
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
std::string teca_vorticity::get_component_0_variable(
    const teca_metadata &request)
{
    std::string comp_0_var = this->component_0_variable;

    if (comp_0_var.empty() &&
        request.has("teca_vorticity::component_0_variable"))
            request.get("teca_vorticity::component_0_variable", comp_0_var);

    return comp_0_var;
}

// --------------------------------------------------------------------------
std::string teca_vorticity::get_component_1_variable(
    const teca_metadata &request)
{
    std::string comp_1_var = this->component_1_variable;

    if (comp_1_var.empty() &&
        request.has("teca_vorticity::component_1_variable"))
            request.get("teca_vorticity::component_1_variable", comp_1_var);

    return comp_1_var;
}

// --------------------------------------------------------------------------
std::string teca_vorticity::get_vorticity_variable(
    const teca_metadata &request)
{
    std::string vort_var = this->vorticity_variable;

    if (vort_var.empty())
    {
        if (request.has("teca_vorticity::vorticity_variable"))
            request.get("teca_vorticity::vorticity_variable", vort_var);
        else
            vort_var = "vorticity";
    }

    return vort_var;
}

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
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the arrays we need to request
    std::string comp_0_var = this->get_component_0_variable(request);
    if (comp_0_var.empty())
    {
        TECA_ERROR("component 0 array was not specified")
        return up_reqs;
    }

    std::string comp_1_var = this->get_component_1_variable(request);
    if (comp_1_var.empty())
    {
        TECA_ERROR("component 0 array was not specified")
        return up_reqs;
    }

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->component_0_variable);
    arrays.insert(this->component_1_variable);

    // capture the array we produce
    arrays.erase(this->get_vorticity_variable(request));

    // update the request
    req.insert("arrays", arrays);

    // send it up
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

    // get component 0 array
    std::string comp_0_var = this->get_component_0_variable(request);

    if (comp_0_var.empty())
    {
        TECA_ERROR("component_0_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array comp_0
        = in_mesh->get_point_arrays()->get(comp_0_var);

    if (!comp_0)
    {
        TECA_ERROR("requested array \"" << comp_0_var << "\" not present.")
        return nullptr;
    }

    // get component 1 array
    std::string comp_1_var = this->get_component_1_variable(request);

    if (comp_1_var.empty())
    {
        TECA_ERROR("component_1_variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array comp_1
        = in_mesh->get_point_arrays()->get(comp_1_var);

    if (!comp_1)
    {
        TECA_ERROR("requested array \"" << comp_1_var << "\" not present.")
        return nullptr;
    }

    // get the input coordinate arrays
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();

    if (!lon || !lat)
    {
        TECA_ERROR("lat lon mesh cooridinates not present.")
        return nullptr;
    }

    // allocate the output array
    p_teca_variant_array vort = comp_0->new_instance();
    vort->resize(comp_0->size());

    // compute vorticity
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        lon.get(), 1,

        const NT1 *p_lon = dynamic_cast<const TT1*>(lon.get())->get();
        const NT1 *p_lat = dynamic_cast<const TT1*>(lat.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            vort.get(), 2,

            const NT2 *p_comp_0 = dynamic_cast<const TT2*>(comp_0.get())->get();
            const NT2 *p_comp_1 = dynamic_cast<const TT2*>(comp_1.get())->get();
            NT2 *p_vort = dynamic_cast<TT2*>(vort.get())->get();

            ::vorticity(p_vort, p_lon, p_lat,
                p_comp_0, p_comp_1, lon->size(), lat->size());
            )
        )

    // create the output mesh, pass everything through, and
    // add the vorticity array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    out_mesh->get_point_arrays()->append(
        this->get_vorticity_variable(request), vort);

    return out_mesh;
}
