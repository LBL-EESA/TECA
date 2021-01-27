#include "teca_binary_segmentation.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <set>
#include <iomanip>

//#define TECA_DEBUG
namespace {

// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void value_threshold(out_t *output, const in_t *input,
    size_t n_vals, in_t low, in_t high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

// predicate for indirect sort
template <typename data_t, typename index_t>
struct indirect_lt
{
    indirect_lt() : p_data(nullptr) {}
    indirect_lt(const data_t *pd) : p_data(pd) {}

    bool operator()(const index_t &a, const index_t &b)
    {
        return p_data[a] < p_data[b];
    }

    const data_t *p_data;
};

template <typename data_t, typename index_t>
struct indirect_gt
{
    indirect_gt() : p_data(nullptr) {}
    indirect_gt(const data_t *pd) : p_data(pd) {}

    bool operator()(const index_t &a, const index_t &b)
    {
        return p_data[a] > p_data[b];
    }

    const data_t *p_data;
};



// Given a vector V of length N, the q-th percentile of V is the value q/100 of
// the way from the minimum to the maximum in a sorted copy of V.

// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void percentile_threshold(out_t *output, const in_t *input,
    unsigned long n_vals, float q_low, float q_high)
{
    // allocate indices and initialize
    using index_t = unsigned long;
    index_t *ids = (index_t*)malloc(n_vals*sizeof(index_t));
    for (index_t i = 0; i < n_vals; ++i)
        ids[i] = i;

    // cut points are locations of values bounding desired percentiles in the
    // sorted data
    index_t n_vals_m1 = n_vals - 1;

    // low percentile is bound from below by value at low_cut
    double tmp = n_vals_m1 * (q_low/100.f);
    index_t low_cut = index_t(tmp);
    double t_low = tmp - low_cut;

    // high percentile is bound from above by value at high_cut+1
    tmp = n_vals_m1 * (q_high/100.f);
    index_t high_cut = index_t(tmp);
    double t_high = tmp - high_cut;

    // compute 4 indices needed for percentile calcultion
    index_t low_cut_p1 = low_cut+1;
    index_t high_cut_p1 = std::min(high_cut+1, n_vals_m1);
    index_t *ids_pn_vals = ids+n_vals;

    // use an indirect comparison that leaves the input data unmodified
    indirect_lt<in_t,index_t>  comp(input);

    // find 2 indices needed for low percentile calc
    std::nth_element(ids, ids+low_cut, ids_pn_vals, comp);
    double y0 = input[ids[low_cut]];

    std::nth_element(ids, ids+low_cut_p1, ids_pn_vals, comp);
    double y1 = input[ids[low_cut_p1]];

    // compute low percetile
    double low_percentile = (y1 - y0)*t_low + y0;

    // find 2 indices needed for the high percentile calc
    std::nth_element(ids, ids+high_cut, ids_pn_vals, comp);
    y0 = input[ids[high_cut]];

    std::nth_element(ids, ids+high_cut_p1, ids_pn_vals, comp);
    y1 = input[ids[high_cut_p1]];

    // compute high percentile
    double high_percentile = (y1 - y0)*t_high + y0;

    /*std::cerr << q_low << "th percentile is " <<  std::setprecision(10) << low_percentile << std::endl
        << q_high << "th percentile is " <<  std::setprecision(9) << high_percentile << std::endl;*/

    // apply thresholds
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low_percentile) && (input[i] <= high_percentile)) ? 1 : 0;

    free(ids);
}

};

// --------------------------------------------------------------------------
teca_binary_segmentation::teca_binary_segmentation() :
    segmentation_variable(""), threshold_variable(""),
    low_threshold_value(std::numeric_limits<double>::lowest()),
    high_threshold_value(std::numeric_limits<double>::max()),
    threshold_mode(BY_VALUE)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_binary_segmentation::~teca_binary_segmentation()
{}

// --------------------------------------------------------------------------
int teca_binary_segmentation::get_segmentation_variable(
    std::string &segmentation_var)
{
    if (this->segmentation_variable.empty())
    {
        std::string threshold_var;
        if (this->get_threshold_variable(threshold_var))
            return -1;

        segmentation_var = threshold_var + "_segments";
    }
    else
    {
        segmentation_var = this->segmentation_variable;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_binary_segmentation::get_threshold_variable(
    std::string &threshold_var)
{
    if (this->threshold_variable.empty())
    {
        TECA_ERROR("Threshold variable is not set")
        return -1;
    }

    threshold_var = this->threshold_variable;
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_binary_segmentation::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_binary_segmentation::get_output_metadata" << endl;
#endif
    (void) port;

    if (this->threshold_variable.empty())
    {
        TECA_ERROR("a threshold_variable has not been set")
        return teca_metadata();
    }

    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);

    // add to the list of available variables
    teca_metadata md = input_md[0];
    md.append("variables", segmentation_var);

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    md.get("attributes", attributes);

    std::ostringstream oss;
    oss << "a binary mask non-zero where " << this->low_threshold_value
        << (this->threshold_mode == BY_VALUE ? "" : "th percentile")
        << " <= " << this->threshold_variable << " <= "
        << (this->threshold_mode == BY_VALUE ? "" : "th percentile")
        << this->high_threshold_value;

    teca_array_attributes default_atts(
        teca_variant_array_code<char>::get(),
        teca_array_attributes::point_centering,
        0, "unitless", segmentation_var, oss.str());

    // start with user provided attributes, provide default values
    // where user attributes are missing
    teca_metadata seg_atts(this->segmentation_variable_attributes);
    default_atts.merge_to(seg_atts);

    attributes.set(segmentation_var, seg_atts);
    md.set("attributes", attributes);

    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_binary_segmentation::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_binary_segmentation::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string threshold_var;
    if (this->get_threshold_variable(threshold_var))
    {
        TECA_ERROR("A threshold variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(threshold_var);

    // remove fromt the request what we generate
    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);
    arrays.erase(segmentation_var);

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_binary_segmentation::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_binary_segmentation::execute" << endl;
#endif
    (void)port;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_ERROR("empty input, or not a cartesian_mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string threshold_var;
    if (this->get_threshold_variable(threshold_var))
    {
        TECA_ERROR("A threshold variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array input_array
        = out_mesh->get_point_arrays()->get(threshold_var);
    if (!input_array)
    {
        TECA_ERROR("threshold variable \"" << threshold_var
            << "\" is not in the input")
        return nullptr;
    }

    // get threshold values
    double low = this->low_threshold_value;
    if (low == std::numeric_limits<double>::lowest()
        && request.has("low_threshold_value"))
        request.get("low_threshold_value", low);

    double high = this->high_threshold_value;
    if (high == std::numeric_limits<double>::max()
        && request.has("high_threshold_value"))
        request.get("high_threshold_value", high);

    // validate the threshold values
    if (this->threshold_mode == BY_PERCENTILE)
    {
        if (low == std::numeric_limits<double>::lowest())
            low = 0.0;

        if (high == std::numeric_limits<double>::max())
            high = 100.0;

        if ((low < 0.0) || (high > 100.0))
        {
            TECA_ERROR("The threshold values are " << low << ", " << high << ". "
              "In percentile mode the threshold values must be between 0 and 100")
            return nullptr;
        }
    }

    // do segmentation
    size_t n_elem = input_array->size();
    p_teca_char_array segmentation =
        teca_char_array::New(n_elem);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        input_array.get(),
        const NT *p_in = static_cast<TT*>(input_array.get())->get();
        char *p_seg = segmentation->get();

        if (this->threshold_mode == BY_VALUE)
        {
            ::value_threshold(p_seg, p_in, n_elem,
               static_cast<NT>(low), static_cast<NT>(high));
        }
        else if  (this->threshold_mode == BY_PERCENTILE)
        {
            ::percentile_threshold(p_seg, p_in, n_elem,
                static_cast<NT>(low), static_cast<NT>(high));
        }
        else
        {
            TECA_ERROR("Invalid threshold mode")
            return nullptr;
        }
        )

    // put segmentation in output
    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);
    out_mesh->get_point_arrays()->set(segmentation_var, segmentation);

    teca_metadata &out_metadata = out_mesh->get_metadata();
    out_metadata.set("low_threshold_value", low);
    out_metadata.set("high_threshold_value", high);

    return out_mesh;
}
