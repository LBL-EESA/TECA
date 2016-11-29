#include "teca_binary_segmentation.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>

using std::deque;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG
namespace {
// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void threshold(out_t *output, const in_t *input,
    size_t n_vals, in_t low, in_t high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}
};

// --------------------------------------------------------------------------
teca_binary_segmentation::teca_binary_segmentation() :
    segmentation_variable(""), threshold_variable(""),
    low_threshold_value(std::numeric_limits<double>::lowest()),
    high_threshold_value(std::numeric_limits<double>::max())
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_binary_segmentation::~teca_binary_segmentation()
{}

// --------------------------------------------------------------------------
std::string teca_binary_segmentation::get_segmentation_variable(
    const teca_metadata &request)
{
    std::string segmentation_var = this->segmentation_variable;
    if (segmentation_var.empty())
    {
        if (request.has("teca_binary_segmentation::segmentation_variable"))
            request.get("teca_binary_segmentation::segmentation_variable", segmentation_var);
        else if (this->threshold_variable.empty())
            segmentation_var = "segments";
        else
            segmentation_var = this->threshold_variable + "segments";
    }
    return segmentation_var;
}


// --------------------------------------------------------------------------
std::string teca_binary_segmentation::get_threshold_variable(
    const teca_metadata &request)
{
    std::string threshold_var = this->threshold_variable;

    if (threshold_var.empty() &&
        request.has("teca_binary_segmentation::threshold_variable"))
            request.get("teca_binary_segmentation::threshold_variable",
                threshold_var);

    return threshold_var;
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


    std::string segmentation_var = this->segmentation_variable;
    if (segmentation_var.empty())
    {
        if (this->threshold_variable.empty())
            segmentation_var = "segmentation";
        else
            segmentation_var = this->threshold_variable + "segmentation";
    }

    teca_metadata md = input_md[0];
    md.append("variables", segmentation_var);
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

    vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string threshold_var = this->get_threshold_variable(request);
    if (threshold_var.empty())
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
    std::string segmentation_var = this->get_segmentation_variable(request);
    arrays.erase(segmentation_var);

    req.insert("arrays", arrays);

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
    std::string threshold_var = this->get_threshold_variable(request);
    if (threshold_var.empty())
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
        && request.has("teca_binary_segmentation::low_threshold_value"))
        request.get("teca_binary_segmentation::low_threshold_value", low);

    double high = this->high_threshold_value;
    if (high == std::numeric_limits<double>::max()
        && request.has("teca_binary_segmentation::high_threshold_value"))
        request.get("teca_binary_segmentation::high_threshold_value", high);

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    // do segmentation and segmentation
    size_t n_elem = input_array->size();
    p_teca_unsigned_int_array segmentation =
        teca_unsigned_int_array::New(n_elem);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        input_array.get(),
        const NT *p_in = static_cast<TT*>(input_array.get())->get();
        unsigned int *p_seg = segmentation->get();

        ::threshold(p_seg, p_in, n_elem,
            static_cast<NT>(low), static_cast<NT>(high));
        )

    // put segmentation in output
    std::string segmentation_var = this->get_segmentation_variable(request);
    out_mesh->get_point_arrays()->set(segmentation_var, segmentation);

    return out_mesh;
}
