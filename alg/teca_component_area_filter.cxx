#include "teca_component_area_filter.h"

#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_string_util.h"

#include <iostream>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

#if defined(TECA_HAS_CUDA)

#include "teca_cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda_impl {

/** Apply the area threshold and mask out labels that are outside.
 * This requires 2*n_labels of shared memory.
 * @param [in] low_area_threshold blobs with area below this value are masked
 * @param [in] high_area_threshold blobs with area above this value are masked
 * @param [in] mask_value the value to assign to blobs outside of the thresholds
 * @param [in] labels_in the labeled image/volume
 * @param [in] n_elem the total number of elements in the image/volume
 * @param [in] areas_in the list of blob areas one per blob
 * @param [in] n_labels the number of blobs
 * @param [out] labels_out the remaining set of lables after the threshold has been applied
 * @param [out] areas_out the areas of the remaining blobs
 * @param [out] label_ids_out the set of labels
 * @param [out] n_labels_out the number of remaining blobs
 */
template <typename label_t, typename area_t>
__global__
void mask_area(area_t low, area_t high, label_t mask_value,
    const label_t * __restrict__ labels_in, size_t n_elem,
    const label_t * __restrict__ ids_in, const area_t * __restrict__ areas_in, size_t n_labels,
    label_t * __restrict__ labels_out, label_t * __restrict__ ids_out, area_t * __restrict__ areas_out,
    size_t * __restrict__ n_labels_out)
{
    auto smem = teca_cuda_util::shared_memory_proxy<label_t>();

    label_t *mask_map = smem;
    label_t *id_map = smem + n_labels;

    // initialize the map in shared memory
    for (int q = threadIdx.x; q < n_labels; ++q)
    {
        area_t A_q = areas_in[q];
        bool keep = !((areas_in[q] < low) || (areas_in[q] > high));
        mask_map[q] = keep ? 1 : 0;
        id_map[q] = keep ? ids_in[q] : mask_value;
    }

    // wait for all threads to complete initialization
    __syncthreads();

    // apply the mask
    size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t q = tidx; q < n_elem; q += stride)
    {
        labels_out[q] = id_map[labels_in[q]];
    }

    // construct the output
    if (tidx == 0)
    {
        int oid = 0;
        for (int q = 0; q < n_labels; ++q)
        {
            if (mask_map[q])
            {
                ids_out[oid] = ids_in[q];
                areas_out[oid] = areas_in[q];
                ++oid;
            }
        }
        *n_labels_out = oid;
    }
}
}
#endif

namespace host_impl {

// this constructs a map from input label id to the output label id, the list
// of ids that survive and their respective areas. if the area of the label is
// outside of the range it will be replaced in the output.
template <typename label_t, typename area_t, typename container_t>
void build_label_map(const label_t *comp_ids, const area_t *areas,
    size_t n, double low_area_threshold, double high_area_threshold,
    label_t mask_value, container_t &label_map,
    label_t *ids_out, area_t *areas_out, size_t &n_labels_out)
{
    n_labels_out = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if ((areas[i] < low_area_threshold) || (areas[i] > high_area_threshold))
        {
            // outside the range, mask this label
            label_map[comp_ids[i]] = mask_value;
        }
        else
        {
            // inside the range, pass it through
            label_map[comp_ids[i]] = comp_ids[i];

            ids_out[n_labels_out] = comp_ids[i];
            areas_out[n_labels_out] = areas[i];

            ++n_labels_out;
        }
    }
}

// visit every point in the data, apply the map. The map is such that labels
// ouside of the specified range are replaced
template <typename label_t, typename container_t>
void apply_label_map(label_t * __restrict__ labels,
    const label_t * __restrict__ labels_in, container_t &label_map, size_t n)
{
    for (unsigned long i = 0; i < n; ++i)
        labels[i] = label_map[labels_in[i]];
}
}



// --------------------------------------------------------------------------
teca_component_area_filter::teca_component_area_filter() :
    component_variable(""), number_of_components_key("number_of_components"),
    component_ids_key("component_ids"), component_area_key("component_area"),
    mask_value(-1), low_area_threshold(std::numeric_limits<double>::lowest()),
    high_area_threshold(std::numeric_limits<double>::max()),
    variable_postfix(""), contiguous_component_ids(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_component_area_filter::~teca_component_area_filter()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_component_area_filter::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_component_area_filter":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_variable,
            "name of the varibale containing connected component labeling")
        TECA_POPTS_GET(std::string, prefix, number_of_components_key,
            "name of the key that contains the number of components")
        TECA_POPTS_GET(std::string, prefix, component_ids_key,
            "name of the key that contains the list of component ids")
        TECA_POPTS_GET(std::string, prefix, component_area_key,
            "name of the key that contains the list of component areas")
        TECA_POPTS_GET(int, prefix, mask_value,
            "components with area outside of the range will be replaced "
            "by this label value")
        TECA_POPTS_GET(double, prefix, low_area_threshold,
            "set the lower end of the range of areas to pass through. "
            "components smaller than this are masked out.")
        TECA_POPTS_GET(double, prefix, high_area_threshold,
            "set the higher end of the range of areas to pass through. "
            "components larger than this are masked out.")
        TECA_POPTS_GET(std::string, prefix, variable_postfix,
            "set a string that will be appended to variable names and "
            "metadata keys in the filter's output")
        TECA_POPTS_GET(int, prefix, contiguous_component_ids,
            "when the region label ids start at 0 and are consecutive "
            "this flag enables use of an optimization")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_component_area_filter::set_properties(const std::string &prefix,
    variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, component_variable)
    TECA_POPTS_SET(opts, std::string, prefix, number_of_components_key)
    TECA_POPTS_SET(opts, std::string, prefix, component_ids_key)
    TECA_POPTS_SET(opts, std::string, prefix, component_area_key)
    TECA_POPTS_SET(opts, int, prefix, mask_value)
    TECA_POPTS_SET(opts, double, prefix, low_area_threshold)
    TECA_POPTS_SET(opts, double, prefix, high_area_threshold)
    TECA_POPTS_SET(opts, std::string, prefix, variable_postfix)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_component_area_filter::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::get_output_metadata" << endl;
#endif
    (void) port;

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    const std::string &var_postfix = this->variable_postfix;
    if (!var_postfix.empty())
    {
        std::string component_var = this->component_variable;
        out_md.append("variables", component_var + var_postfix);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_component_area_filter::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    if (this->component_variable.empty())
    {
        TECA_FATAL_ERROR("The component variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(this->component_variable);

    // remove the arrays we produce if the post-fix is set,
    // and replace it with the actual requested array.
    const std::string &var_postfix = this->variable_postfix;
    if (!var_postfix.empty())
    {
        teca_string_util::remove_postfix(arrays, var_postfix);
    }

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_component_area_filter::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_component_area_filter::execute" << endl;
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
    if (this->component_variable.empty())
    {
        TECA_FATAL_ERROR("The component variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array labels_in
        = out_mesh->get_point_arrays()->get(this->component_variable);

    if (!labels_in)
    {
        TECA_FATAL_ERROR("labels variable \"" << this->component_variable
            << "\" is not in the input")
        return nullptr;
    }

    // get the list of component ids
    teca_metadata &in_metadata =
        const_cast<teca_metadata&>(in_mesh->get_metadata());

    const_p_teca_variant_array ids_in
        = in_metadata.get(this->component_ids_key);

    if (!ids_in)
    {
        TECA_FATAL_ERROR("Metadata missing component ids")
        return nullptr;
    }

    size_t n_ids_in = ids_in->size();

    // get the list of component areas
    const_p_teca_variant_array areas_in
        = in_metadata.get(this->component_area_key);

    if (!areas_in)
    {
        TECA_FATAL_ERROR("Metadata missing component areas")
        return nullptr;
    }

    // get the mask value
    long long mask_value = this->mask_value;
    if (this->mask_value == -1)
    {
        if (in_metadata.get("background_id", mask_value))
        {
            TECA_FATAL_ERROR("Metadata is missing the key \"background_id\". "
                "One should specify it via the \"mask_value\" algorithm "
                "property")
            return nullptr;
        }
    }

    // get threshold values. these may be passed from a downstream algorithm
    // as in the case of the TECA BARD
    double low_val = this->low_area_threshold;
    if (low_val == std::numeric_limits<double>::lowest()
        && request.has("low_area_threshold"))
        request.get("low_area_threshold", low_val);

    double high_val = this->high_area_threshold;
    if (high_val == std::numeric_limits<double>::max()
        && request.has("high_area_threshold"))
        request.get("high_area_threshold", high_val);

    // determine if the calculation runs on the cpu or gpu
    allocator alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);

    // our GPU implementation requires contiguous component ids
    if (device_id >= 0)
    {
        if (!this->contiguous_component_ids)
        {
            TECA_WARNING("Requested executiong on device " << device_id
                << ". Execution moved to host because of non-contiguous"
                " component ids.")
            device_id = -1;
        }
        else
        {
            alloc = allocator::cuda_async;
        }
    }
#endif

    // allocate the array to store the output with labels outside the requested
    // range removed.
    size_t n_elem = labels_in->size();
    p_teca_variant_array labels_out = labels_in->new_instance(n_elem, alloc);

    // pass to the output
    std::string labels_var_postfix = this->component_variable + this->variable_postfix;
    out_mesh->get_point_arrays()->set(labels_var_postfix, labels_out);

    // get the output metadata to add results to after the filter is applied
    teca_metadata &out_metadata = out_mesh->get_metadata();

    size_t n_ids_out = 0;
    p_teca_variant_array ids_out;
    p_teca_variant_array areas_out;

    // apply the filter
    NESTED_VARIANT_ARRAY_DISPATCH_I(
        labels_out.get(), _LABEL,

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            areas_in.get(), _AREA,

            // get or allocate the outputs
            auto [p_labels_out] = data<TT_LABEL>(labels_out);

            NT_LABEL *p_ids_out = nullptr;
            std::tie(ids_out, p_ids_out) = teca_variant_array_util::New<TT_LABEL>(n_ids_in, alloc);

            NT_AREA *p_areas_out = nullptr;
            std::tie(areas_out, p_areas_out) = teca_variant_array_util::New<TT_AREA>(n_ids_in, alloc);

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                if (teca_cuda_util::set_device(device_id))
                    return nullptr;

                // get the inputs
                auto [sp_labels_in, p_labels_in] = get_cuda_accessible<CTT_LABEL>(labels_in);
                auto [sp_ids_in, p_ids_in] = get_cuda_accessible<CTT_LABEL>(ids_in);
                auto [sp_areas_in, p_areas_in] = get_cuda_accessible<CTT_AREA>(areas_in);

                size_t n_ids_in = ids_in->size();

                hamr::buffer<size_t> n_ids_out_buf(hamr::buffer_allocator::cuda_async, size_t(1), size_t(0));

                int blkDim = 128;
                int gridDim = n_elem / blkDim + ( n_elem % blkDim ? 1 : 0 );
                int sharedMem = 2*n_ids_in*sizeof(NT_LABEL);
                cudaStream_t strm = cudaStreamPerThread;

                // apply the mask
                cuda_impl::mask_area<<<gridDim, blkDim, sharedMem, strm>>>
                    (NT_AREA(low_val), NT_AREA(high_val), NT_LABEL(mask_value),
                    p_labels_in, n_elem, p_ids_in, p_areas_in, n_ids_in,
                    p_labels_out, p_ids_out, p_areas_out, n_ids_out_buf.data());

                // correct the size of the output
                n_ids_out_buf.get(0, &n_ids_out, 0, 1);
            }
            else
            {
#endif
                // get the inputs
                auto [sp_labels_in, p_labels_in] = get_host_accessible<CTT_LABEL>(labels_in);
                auto [sp_ids_in, p_ids_in] = get_host_accessible<CTT_LABEL>(ids_in);
                auto [sp_areas_in, p_areas_in] = get_host_accessible<CTT_AREA>(areas_in);

                sync_host_access_any(labels_in, ids_in, areas_in);

                // if we have labels with small values we can speed the calculation by
                // using a contiguous buffer to hold the map. otherwise we need to
                // use an associative container
                if (this->contiguous_component_ids)
                {
                    // find the max label id, used to size the map buffer
                    NT_LABEL max_id = std::numeric_limits<NT_LABEL>::lowest();
                    for (unsigned int i = 0; i < n_ids_in; ++i)
                        max_id = std::max(max_id, p_ids_in[i]);

                    // allocate the map
                    std::vector<NT_LABEL> label_map(max_id+1, NT_LABEL(mask_value));

                    // construct the map from input label to output label.
                    // removing a lable from the output ammounts to applying
                    // the mask value to the labels
                    host_impl::build_label_map(p_ids_in, p_areas_in, n_ids_in,
                            low_val, high_val, NT_LABEL(mask_value),
                            label_map, p_ids_out, p_areas_out, n_ids_out);

                    // use the map to mask out removed labels
                    host_impl::apply_label_map(p_labels_out, p_labels_in, label_map, n_elem);
                }
                else
                {
                    std::map<NT_LABEL, NT_LABEL> label_map;

                    // construct the map from input label to output label.
                    // removing a lable from the output ammounts to applying
                    // the mask value to the labels
                    host_impl::build_label_map(p_ids_in, p_areas_in, n_ids_in,
                            low_val, high_val, NT_LABEL(mask_value),
                            label_map, p_ids_out, p_areas_out, n_ids_out);

                    // use the map to mask out removed labels
                    host_impl::apply_label_map(p_labels_out, p_labels_in, label_map, n_elem);
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
            )
        )

    // correct the size of the output
    ids_out->resize(n_ids_out);
    areas_out->resize(n_ids_out);

    // pass the updated set of component ids and their coresponding areas
    // to the output
    out_metadata.set(this->number_of_components_key + this->variable_postfix, n_ids_out);
    out_metadata.set(this->component_ids_key + this->variable_postfix, ids_out);
    out_metadata.set(this->component_area_key + this->variable_postfix, areas_out);
    out_metadata.set("background_id" + this->variable_postfix, mask_value);

    // pass the threshold values used
    out_metadata.set("low_area_threshold_km", low_val);
    out_metadata.set("high_area_threshold_km", high_val);


    return out_mesh;
}
