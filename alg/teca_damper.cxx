#include "teca_damper.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>
#include <cmath>
#include <complex.h>

using std::deque;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG
namespace {

// get the filter ready to be applied in the next steps 
template <typename coord_t>
void get_lat_filter(
    coord_t *filter, const coord_t *lat_input, size_t n_lat_vals,
    double filter_lat_width)
{
    double filter_lat_width_squared = filter_lat_width * filter_lat_width;
    for (size_t i = 0; i < n_lat_vals; ++i)
    {
        coord_t lat_in_squared = lat_input[i] * lat_input[i];
        filter[i] = 1.0 - exp(-1.0 * lat_in_squared/(2 * filter_lat_width_squared));
    }
}

// damp the input array using inverted gaussian 
template <typename num_t, typename coord_t>
void apply_damper(
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
teca_damper::teca_damper() :
    filter_lat_width_value(std::numeric_limits<double>::quiet_NaN()),
    post_fix("_damped")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_damper::~teca_damper()
{}

// --------------------------------------------------------------------------
std::vector<std::string> teca_damper::get_damper_variables(
    const teca_metadata &request)
{
    std::vector<std::string> damper_vars = this->damper_variables;
    if (damper_vars.empty() &&
        request.has("teca_damper::damper_variables"))
            request.get("teca_damper::damper_variables", damper_vars);
        
    return damper_vars;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_damper::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_damper::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::vector<std::string> damper_vars = this->get_damper_variables(request);
    if (damper_vars.empty())
    {
        TECA_ERROR("damper variables were not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(damper_vars.begin(), damper_vars.end());

    req.insert("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_damper::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_damper::execute" << endl;
#endif
    (void)port;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
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
    std::vector<std::string> damper_vars = this->get_damper_variables(request);
    if (damper_vars.empty())
    {
        TECA_ERROR("A damper variable was not specified")
        return nullptr;
    }

    // get threshold values
    double filter_lat_width_val = this->filter_lat_width_value;
    std::string post_fix = this->post_fix;

    if (std::isnan(filter_lat_width_val)
        && request.has("teca_damper::filter_lat_width_value"))
        request.get("teca_damper::filter_lat_width_value", filter_lat_width_val);

    if (std::isnan(filter_lat_width_val))
    {
        TECA_ERROR("A filter_lat_width (delta-Y) value was not specified")
        return nullptr;
    }

    // get the coordinate axes
    const_p_teca_variant_array yc = in_mesh->get_y_coordinates();
    const_p_teca_variant_array xc = in_mesh->get_x_coordinates();

    size_t n_lat = yc->size();
    size_t n_lon = xc->size();

    p_teca_variant_array filter_array = yc->new_instance(n_lat);

    // Get the gaussian filter
    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        filter_array.get(),
        _COORD,

        const NT_COORD *p_yc = static_cast<const TT_COORD*>(yc.get())->get();

        NT_COORD *p_filter_array = static_cast<TT_COORD*>(filter_array.get())->get();

        ::get_lat_filter(p_filter_array, p_yc, n_lat, filter_lat_width_val);

        size_t n_arrays = damper_vars.size();
        for (size_t i = 0; i < n_arrays; ++i)
        {
            // get the input array
            const_p_teca_variant_array input_array
                = out_mesh->get_point_arrays()->get(damper_vars[i]);
            if (!input_array)
            {
                TECA_ERROR("damper variable \"" << damper_vars[i]
                    << "\" is not in the input")
                return nullptr;
            }

            // apply the gaussian damper 
            size_t n_elem = input_array->size();
            p_teca_variant_array damped_array = input_array->new_instance(n_elem);

            TEMPLATE_DISPATCH(teca_variant_array_impl,
                damped_array.get(),

                const NT *p_in = static_cast<const TT*>(input_array.get())->get();
                NT *p_damped_array = static_cast<TT*>(damped_array.get())->get();

                ::apply_damper(p_damped_array, p_in, p_filter_array, 
                    n_lat, n_lon);
            )

            // set the damped array in the output
            std::string damper_var_post_fix = damper_vars[i] + post_fix;
            out_mesh->get_point_arrays()->set(damper_var_post_fix, damped_array);
        }
    )

    return out_mesh;
}

