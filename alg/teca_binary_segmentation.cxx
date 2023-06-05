#include "teca_binary_segmentation.h"
#include "teca_binary_segmentation_internals.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_cartesian_mesh.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <set>
#include <iomanip>

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
        TECA_FATAL_ERROR("Threshold variable is not set")
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
        TECA_FATAL_ERROR("Threshold variable is not set")
        return {};
    }

    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);

    // add to the list of available variables
    teca_metadata md = input_md[0];
    md.append("variables", segmentation_var);

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    md.get("attributes", attributes);

    teca_metadata thresh_atts;
    attributes.get(this->threshold_variable, thresh_atts);

    auto dim_active = teca_array_attributes::xyzt_active();
    thresh_atts.get("mesh_dim_active", dim_active);

    std::ostringstream oss;
    oss << "a binary mask non-zero where " << this->low_threshold_value
        << (this->threshold_mode == BY_VALUE ? "" : "th percentile")
        << " <= " << this->threshold_variable << " <= "
        << (this->threshold_mode == BY_VALUE ? "" : "th percentile")
        << this->high_threshold_value;

    teca_array_attributes default_atts(
        teca_variant_array_code<char>::get(),
        teca_array_attributes::point_centering, 0, dim_active,
        "unitless", segmentation_var, oss.str());

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
        TECA_FATAL_ERROR("A threshold variable was not specified")
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
    std::cerr << teca_parallel_id()
        << "teca_binary_segmentation::execute" << std::endl;
#endif
    (void)port;

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_FATAL_ERROR("empty input, or not a cartesian_mesh")
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
        TECA_FATAL_ERROR("A threshold variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array input_array
        = out_mesh->get_point_arrays()->get(threshold_var);
    if (!input_array)
    {
        TECA_FATAL_ERROR("threshold variable \"" << threshold_var
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
            TECA_FATAL_ERROR("The threshold values are " << low << ", " << high << ". "
              "In percentile mode the threshold values must be between 0 and 100")
            return nullptr;
        }
    }

    // do segmentation
    p_teca_variant_array segmentation;

    int device_id = -1;
    request.get("device_id", device_id);

    if (device_id >= 0)
    {
#ifdef TECA_DEBUG
        std::cerr << "executing on CUDA device " << device_id << std::endl;
#endif
        if (teca_binary_segmentation_internals::cuda_dispatch(device_id,
            segmentation, input_array, this->threshold_mode, low, high))
        {
            TECA_FATAL_ERROR("Failed to segment on the GPU")
            return nullptr;
        }
    }
    else
    {
#ifdef TECA_DEBUG
        std::cerr << "executing on the CPU" << std::endl;
#endif
        if (teca_binary_segmentation_internals::cpu_dispatch(segmentation,
            input_array, this->threshold_mode, low, high))
        {
            TECA_FATAL_ERROR("Failed to segment on the CPU")
            return nullptr;
        }
    }

    // put segmentation in output
    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);
    out_mesh->get_point_arrays()->set(segmentation_var, segmentation);

    teca_metadata &out_metadata = out_mesh->get_metadata();
    out_metadata.set("low_threshold_value", low);
    out_metadata.set("high_threshold_value", high);

    return out_mesh;
}
