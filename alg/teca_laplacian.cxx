#include "teca_laplacian.h"

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
using std::tan;

//#define TECA_DEBUG

namespace {

template <typename num_t>
constexpr num_t deg_to_rad() { return num_t(M_PI)/num_t(180); }

template <typename num_t>
constexpr num_t earth_radius() { return num_t(6371.0e3); }

// compute the laplacian. This  assumes fixed mesh spacing. Here we add periodic
// bc in lon and apply unit stride vector optimization strategy to loops
template <typename num_t, typename pt_t>
void laplacian(num_t *w, const pt_t *lon, const pt_t *lat,
    const num_t *f, unsigned long n_lon,
    unsigned long n_lat, bool periodic_lon=true)
{
    size_t n_bytes = n_lat*sizeof(num_t);
    num_t *delta_lon_sq = static_cast<num_t*>(malloc(n_bytes));

    // delta lon squared as a function of latitude
    num_t d_lon = (lon[1] - lon[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();
    // tan(lat)
    num_t *tan_lat = static_cast<num_t*>(malloc(n_bytes)); 
    for (unsigned long j = 0; j < n_lat; ++j)
    {
        delta_lon_sq[j] = pow(d_lon * cos(lat[j] * deg_to_rad<num_t>()),2);
    	tan_lat[j] = tan(lat[j] * deg_to_rad<num_t>());
    }

    // delta lat squared
    num_t delta_v = (lat[1] - lat[0]) * deg_to_rad<num_t>() * earth_radius<num_t>();
    num_t dlat = num_t(2)*delta_v;
    num_t dlat_sq = delta_v*delta_v;
    dlat *= earth_radius<num_t>(); // scale dlat by R for the tan term

    unsigned long max_i = n_lon - 1;
    unsigned long max_j = n_lat - 1;

    // laplacian
    for (unsigned long j = 1; j < max_j; ++j)
    {
	// set the current row in the u/v/w arrays
        unsigned long jj = j*n_lon;
	/* 
	 * The following f_* variables describe the field
	 * f in a grid oriented fashion:
	 *
	 *	f_ipjm	f_ipj	f_ipjp
	 *
	 *	f_ijm	f_ji	f_ijp
	 *
	 *	f_imjm	f_imj	f_imjp
	 * 
	 * The 'j' direction represents longitude, the
	 * 'i' direciton represents latitude. 
	 *
	 * Note: The laplacian represented here uses the chain
	 * rule to separate the (1/cos(lat)*d(cos(lat)*df/dlat)/dlat 
	 * term into two terms.
	 *
	 */
	// Set array pointer locations so that index 'i' refers to the
	// shifted location in all variables
        const num_t *f_ij = f + jj;          // i,j
        const num_t *f_ipj = f + jj + n_lon; // i+1, j
        const num_t *f_imj = f + jj - n_lon; // i-1, j
        const num_t *f_ijp = f + jj + 1;     // i,   j + 1
        const num_t *f_ijm = f + jj - 1;     // i,   j - 1
	
	// set the pointer index for the output field w
	// ... this is index i,j
        num_t *ww = w + jj;
	// create a dummy variable for u**2 
        num_t dlon_sq = delta_lon_sq[j];

        for (unsigned long i = 1; i < max_i; ++i)
        {
	    // calculate the laplacian in spherical coordinates, assuming
	    // constant radius R.
            ww[i] = (f_imj[i] - num_t(2)*f_ij[i] + f_ipj[i])/dlat_sq - 
		    tan_lat[j]*(f_ipj[i]-f_imj[i])/dlat + 
                    (f_ijm[i] - num_t(2)*f_ij[i] + f_ijp[i])/dlon_sq;
        }
    }

    if (periodic_lon)
    {
        // periodic in longitude; leftmost boundary
        for (unsigned long j = 1; j < max_j; ++j)
        {
	    // set the current row in the u/v/w arrays
            unsigned long jj = j*n_lon;
	    // Set array pointer locations so that index 'i' refers to the
	    // shifted location in all variables
            const num_t *f_ij = f + jj;          // i,j
            const num_t *f_ipj = f + jj + n_lon; // i+1, j
            const num_t *f_imj = f + jj - n_lon; // i-1, j
            const num_t *f_ijp = f + jj + 1;     // i,   j + 1
            const num_t *f_ijm = f + jj - max_i; // i,   j - 1

	    // set the pointer index for the output field w
	    // ... this is index i,j
            num_t *ww = w + jj;
	    // create a dummy variable for u**2 
            num_t dlon_sq = delta_lon_sq[j];

	    // calculate the laplacian in spherical coordinates, assuming
	    // constant radius R.
            ww[0] = (f_imj[0] - num_t(2)*f_ij[0] + f_ipj[0])/dlat_sq - 
		    tan_lat[j]*(f_ipj[0]-f_imj[0])/dlat + 
                    (f_ijm[0] - num_t(2)*f_ij[0] + f_ijp[0])/dlon_sq;
        }

        // periodic in longitude; rightmost boundary
        for (unsigned long j = 1; j < max_j; ++j)
        {
	    // set the current row in the u/v/w arrays
            unsigned long jj = j*n_lon;

	    // Set array pointer locations so that index 'i' refers to the
	    // shifted location in all variables
            const num_t *f_ij = f + jj + max_i;          // i,j
            const num_t *f_ipj = f + jj + max_i + n_lon; // i+1, j
            const num_t *f_imj = f + jj + max_i - n_lon; // i-1, j
            const num_t *f_ijp = f + jj;                 // i,   j + 1
            const num_t *f_ijm = f + jj - max_i;         // i,   j - 1

	    // set the pointer index for the output field w
	    // ... this is index i,j
            num_t *ww = w + jj + max_i;
	    // create a dummy variable for u**2 
            num_t dlon_sq = delta_lon_sq[j];

	    // calculate the laplacian in spherical coordinates, assuming
	    // constant radius R.
            ww[0] = (f_imj[0] - num_t(2)*f_ij[0] + f_ipj[0])/dlat_sq - 
		    tan_lat[j]*(f_ipj[0]-f_imj[0])/dlat + 
                    (f_ijm[0] - num_t(2)*f_ij[0] + f_ijp[0])/dlon_sq;
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

    free(delta_lon_sq);
    free(tan_lat);

    return;
}
};


// --------------------------------------------------------------------------
teca_laplacian::teca_laplacian() :
    component_0_variable(), 
    laplacian_variable("laplacian")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_laplacian::~teca_laplacian()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_laplacian::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_laplacian":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable,
            "array containing the input variable")
        TECA_POPTS_GET(std::string, prefix, laplacian_variable,
            "array to store the computed laplacian in")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_laplacian::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, component_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, laplacian_variable)
}
#endif

// --------------------------------------------------------------------------
std::string teca_laplacian::get_component_0_variable(
    const teca_metadata &request)
{
    std::string comp_0_var = this->component_0_variable;

    if (comp_0_var.empty() &&
        request.has("teca_laplacian::component_0_variable"))
            request.get("teca_laplacian::component_0_variable", comp_0_var);

    return comp_0_var;
}

// --------------------------------------------------------------------------
std::string teca_laplacian::get_laplacian_variable(
    const teca_metadata &request)
{
    std::string lapl_var = this->laplacian_variable;

    if (lapl_var.empty())
    {
        if (request.has("teca_laplacian::laplacian_variable"))
            request.get("teca_laplacian::laplacian_variable", lapl_var);
        else
            lapl_var = "laplacian";
    }

    return lapl_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_laplacian::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_laplacian::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);
    out_md.append("variables", this->laplacian_variable);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_laplacian::get_upstream_request(
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

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->component_0_variable);

    // capture the array we produce
    arrays.erase(this->get_laplacian_variable(request));

    // update the request
    req.insert("arrays", arrays);

    // send it up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_laplacian::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_laplacian::execute" << endl;
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

    // get the input coordinate arrays
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();

    if (!lon || !lat)
    {
        TECA_ERROR("lat lon mesh cooridinates not present.")
        return nullptr;
    }

    // allocate the output array
    p_teca_variant_array lapl = comp_0->new_instance();
    lapl->resize(comp_0->size());

    // compute laplacian
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        lon.get(), 1,

        const NT1 *p_lon = dynamic_cast<const TT1*>(lon.get())->get();
        const NT1 *p_lat = dynamic_cast<const TT1*>(lat.get())->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            lapl.get(), 2,

            const NT2 *p_comp_0 = dynamic_cast<const TT2*>(comp_0.get())->get();
            NT2 *p_lapl = dynamic_cast<TT2*>(lapl.get())->get();

            ::laplacian(p_lapl, p_lon, p_lat,
                p_comp_0, lon->size(), lat->size());
            )
        )

    // create the output mesh, pass everything through, and
    // add the laplacian array
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    out_mesh->get_point_arrays()->append(
        this->get_laplacian_variable(request), lapl);

    return out_mesh;
}
