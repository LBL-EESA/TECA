#include "teca_latitude_damper.h"

#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_string_util.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#include <complex.h>

using std::cerr;
using std::endl;

//#define TECA_DEBUG
namespace {

// get the filter ready to be applied in the next steps
template <typename coord_t>
void get_lat_filter(
    coord_t *filter, const coord_t *lat, size_t n_lat_vals,
    coord_t mu, coord_t sigma)
{
    coord_t two_sigma_sqr = 2.0*sigma*sigma;
    for (size_t i = 0; i < n_lat_vals; ++i)
    {
        coord_t x_min_mu = lat[i] - mu;
        coord_t neg_x_min_mu_sqr = -x_min_mu*x_min_mu;
        filter[i] = coord_t(1) - exp(neg_x_min_mu_sqr/two_sigma_sqr);
    }
}

// damp the input array using inverted gaussian
template <typename num_t, typename coord_t>
void apply_lat_filter(
    num_t *output, const num_t *input, const coord_t *filter,
    size_t n_lat_vals, size_t n_lon_vals)
{
    for (size_t j = 0; j < n_lat_vals; ++j)
    {
        size_t jj = j * n_lon_vals;
        for (size_t i = 0; i < n_lon_vals; ++i)
        {
            output[jj + i] = filter[j] * input[jj + i];
        }
    }
}

};

// --------------------------------------------------------------------------
teca_latitude_damper::teca_latitude_damper() :
    center(std::numeric_limits<double>::quiet_NaN()),
    half_width_at_half_max(std::numeric_limits<double>::quiet_NaN()),
    variable_post_fix("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_latitude_damper::~teca_latitude_damper()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_latitude_damper::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_latitude_damper":prefix));

    opts.add_options()
        TECA_POPTS_GET(double, prefix, center,
            "set the center (mu) for the gaussian filter")
        TECA_POPTS_GET(double, prefix, half_width_at_half_max,
            "set the value of the half width at half maximum (HWHM) "
            "to calculate sigma from: sigma = HWHM/std::sqrt(2.0*std::log(2.0))")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, damped_variables,
            "set the variables that will be damped by the inverted "
            "gaussian filter")
        TECA_POPTS_GET(std::string, prefix, variable_post_fix,
            "set the post-fix that will be attached to the variables "
            "that will be saved in the output")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}
// --------------------------------------------------------------------------
void teca_latitude_damper::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, double, prefix, center)
    TECA_POPTS_SET(opts, double, prefix, half_width_at_half_max)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, damped_variables)
    TECA_POPTS_SET(opts, std::string, prefix, variable_post_fix)
}
#endif

// --------------------------------------------------------------------------
int teca_latitude_damper::get_sigma(const teca_metadata &request, double &sigma)
{
    double hwhm = 0.0;
    if (std::isnan(this->half_width_at_half_max))
    {
        if (request.has("half_width_at_half_max"))
            request.get("half_width_at_half_max", hwhm);
        else
            return -1;
    }
    else
    {
        hwhm = this->half_width_at_half_max;
    }

    sigma = hwhm/std::sqrt(2.0*std::log(2.0));

    return 0;
}

// --------------------------------------------------------------------------
int teca_latitude_damper::get_mu(const teca_metadata &request, double &mu)
{
    if (std::isnan(this->center))
    {
        if (request.has("center"))
            request.get("center", mu);
        else
            return -1;
    }
    else
    {
        mu = this->center;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_latitude_damper::get_damped_variables(std::vector<std::string> &vars)
{
    if (this->damped_variables.empty())
        return -1;
    else
        vars = this->damped_variables;

    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_latitude_damper::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_latitude_damper::get_output_metadata" << endl;
#endif
    (void)port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &var_post_fix = this->variable_post_fix;
    if (!var_post_fix.empty())
    {
        std::vector<std::string> &damped_vars = this->damped_variables;

        size_t n_arrays = damped_vars.size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            out_md.append("variables", damped_vars[i] + var_post_fix);
        }
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_latitude_damper::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_latitude_damper::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;
    teca_metadata req(request);

    // get the name of the array to request
    std::vector<std::string> damped_vars;
    if (this->get_damped_variables(damped_vars))
    {
        TECA_ERROR("No variables to damp specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(damped_vars.begin(), damped_vars.end());

    // Cleaning off the postfix for arrays passed in the pipeline.
    // For ex a down stream could request "foo_damped" then we'd
    // need to request "foo". also remove "foo_damped" from the
    // request.
    const std::string &var_post_fix = this->variable_post_fix;
    if (!var_post_fix.empty())
    {
        teca_string_util::remove_post_fix(arrays, var_post_fix);
    }

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_latitude_damper::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_latitude_damper::execute" << endl;
#endif

    (void)port;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(in_mesh->new_instance());

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array names
    std::vector<std::string> damped_vars;
    if (this->get_damped_variables(damped_vars))
    {
        TECA_ERROR("No variable specified to damp")
        return nullptr;
    }

    // get Gaussian paramters. if none were provided, these are the defaults
    // that will be used.
    double mu = 0.0;
    double sigma = 45.0;

    this->get_mu(request, mu);
    this->get_sigma(request, sigma);

    // get the coordinate axes
    const_p_teca_variant_array lat = in_mesh->get_y_coordinates();
    const_p_teca_variant_array lon = in_mesh->get_x_coordinates();

    size_t n_lat = lat->size();
    size_t n_lon = lon->size();

    p_teca_variant_array filter_array = lat->new_instance(n_lat);

    // Get the gaussian filter
    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        filter_array.get(),
        _COORD,

        const NT_COORD *p_lat = static_cast<const TT_COORD*>(lat.get())->get();

        NT_COORD *filter = (NT_COORD*)malloc(n_lat*sizeof(NT_COORD));
        ::get_lat_filter<NT_COORD>(filter, p_lat, n_lat, mu, sigma);

        size_t n_arrays = damped_vars.size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            // get the input array
            const_p_teca_variant_array input_array
                = out_mesh->get_point_arrays()->get(damped_vars[i]);
            if (!input_array)
            {
                TECA_ERROR("damper variable \"" << damped_vars[i]
                    << "\" is not in the input")
                return nullptr;
            }

            // apply the gaussian damper
            size_t n_elem = input_array->size();
            p_teca_variant_array damped_array = input_array->new_instance(n_elem);

            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                damped_array.get(),
                _DATA,

                const NT_DATA *p_in = static_cast<const TT_DATA*>(input_array.get())->get();
                NT_DATA *p_damped_array = static_cast<TT_DATA*>(damped_array.get())->get();

                ::apply_lat_filter(p_damped_array, p_in, filter, n_lat, n_lon);
            )

            // set the damped array in the output
            std::string out_var_name = damped_vars[i] + this->variable_post_fix;
            out_mesh->get_point_arrays()->set(out_var_name, damped_array);
        }

        free(filter);
    )

    teca_metadata &omd = out_mesh->get_metadata();
    omd.set("gaussian_filter_hwhm", sigma);
    omd.set("gaussian_filter_center_lat", mu);

    return out_mesh;
}

