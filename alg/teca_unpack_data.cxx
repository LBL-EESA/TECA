#include "teca_unpack_data.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

namespace cpu
{
// **************************************************************************
template <typename input_t, typename output_t>
void transform(output_t * __restrict__ p_out, input_t * __restrict__ p_in,
    size_t n, output_t scale, output_t offset)
{
    for (size_t i = 0; i < n; ++i)
        p_out[i] = p_in[i] * scale + offset;
}

// **************************************************************************
template <typename input_t, typename mask_t, typename output_t>
void transform(output_t * __restrict__ p_out, input_t * __restrict__ p_in,
    mask_t * __restrict__ p_mask, size_t n, output_t scale, output_t offset,
    output_t fill)
{
    for (size_t i = 0; i < n; ++i)
        p_out[i] = (p_mask[i] ? p_in[i] * scale + offset : fill);
}

// **************************************************************************
int dispatch(const p_teca_variant_array &in_array,
    const p_teca_variant_array &mask, double scale,
    double offset, p_teca_variant_array &out_array,
    int output_data_type)
{
#if defined(TECA_DEBUG)
    std::cerr << "running on the cpu" << std::endl;
#endif
    // allocate the output
    out_array = teca_variant_array_factory::New(output_data_type);
    if (!out_array)
    {
        TECA_FATAL_ERROR("Failed to allocate the output array")
        return -1;
    }

    unsigned long n_elem = in_array->size();
    out_array->resize(n_elem);

    // transform arrays
    NESTED_VARIANT_ARRAY_DISPATCH(
        in_array.get(), _IN,

        auto [sp_in, p_in] = get_host_accessible<TT_IN>(in_array);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            out_array.get(), _OUT,

            auto [p_out] = data<TT_OUT>(out_array);

            if (mask)
            {
                NESTED_VARIANT_ARRAY_DISPATCH_I(
                    mask.get(), _MASK,

                    auto [sp_mask, p_mask] = get_host_accessible<TT_MASK>(mask);

                    sync_host_access_any(in_array, mask);

                    cpu::transform(p_out, p_in, p_mask,
                        n_elem, NT_OUT(scale), NT_OUT(offset), NT_OUT(1e20));
                    )

            }
            else
            {
                sync_host_access_any(in_array);
                cpu::transform(p_out, p_in, n_elem, NT_OUT(scale), NT_OUT(offset));
            }
            )
        )

    return 0;
}
}

#if defined(TECA_HAS_CUDA)
namespace cuda_gpu
{
// **************************************************************************
template <typename input_t, typename output_t>
__global__
void transform(output_t * __restrict__ p_out, input_t * __restrict__ p_in,
    size_t n_elem, output_t scale, output_t offset)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_out[i] = p_in[i] * scale + offset;
}

// **************************************************************************
template <typename input_t, typename mask_t, typename output_t>
__global__
void transform(output_t * __restrict__ p_out, input_t * __restrict__ p_in,
    mask_t * __restrict__ p_mask, size_t n_elem, output_t scale, output_t offset,
    output_t fill)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_out[i] = (p_mask[i] ? p_in[i] * scale + offset : fill);
}

// **************************************************************************
template <typename input_t, typename mask_t, typename output_t>
int transform(int device_id, output_t *p_out, input_t *p_in,
    mask_t *p_mask, size_t n_elem, output_t scale,
    output_t offset, output_t fill)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // unpack the data
    cudaError_t ierr = cudaSuccess;
    cuda_gpu::transform<<<block_grid,thread_grid>>>(p_out,
        p_in, p_mask, n_elem, scale, offset, fill);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the transform CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// **************************************************************************
template <typename input_t, typename output_t>
int transform(int device_id, output_t *p_out, input_t *p_in,
    size_t n_elem, output_t scale, output_t offset)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // unpack the data
    cudaError_t ierr = cudaSuccess;
    cuda_gpu::transform<<<block_grid,thread_grid>>>(p_out,
        p_in, n_elem, scale, offset);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the transform CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// **************************************************************************
int dispatch(int device_id, const p_teca_variant_array &in_array,
    const p_teca_variant_array &mask, double scale, double offset,
    p_teca_variant_array &out_array, int output_data_type)
{
#if defined(TECA_DEBUG)
    std::cerr << "running on gpu " << device_id  << std::endl;
#endif
    teca_cuda_util::set_device(device_id);

    // allocate the output
    out_array = teca_variant_array_factory::New
        (output_data_type, allocator::cuda_async);

    if (!out_array)
    {
        TECA_FATAL_ERROR("Failed to allocate the output array")
        return -1;
    }

    unsigned long n_elem = in_array->size();
    out_array->resize(n_elem);

    // transform arrays
    NESTED_VARIANT_ARRAY_DISPATCH(
        in_array.get(), _IN,

        auto [sp_in, p_in] = get_cuda_accessible<TT_IN>(in_array);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            out_array.get(), _OUT,

            auto [p_out] = data<TT_OUT>(out_array);

            if (mask)
            {
                NESTED_VARIANT_ARRAY_DISPATCH_I(
                    mask.get(), _MASK,

                    auto [sp_mask, p_mask] = get_cuda_accessible<TT_MASK>(mask);

                    // unpack the data
                    if (cuda_gpu::transform(device_id, p_out, p_in, p_mask,
                        n_elem, NT_OUT(scale), NT_OUT(offset), NT_OUT(1e20)))
                        return -1;
                    )
            }
            else
            {
                // unpack the data
                if (cuda_gpu::transform(device_id, p_out, p_in,
                    n_elem, NT_OUT(scale), NT_OUT(offset)))
                    return -1;
            }
            )
        )

    return 0;
}
}
#endif

// --------------------------------------------------------------------------
teca_unpack_data::teca_unpack_data() :
    output_data_type(teca_variant_array_code<float>::get())
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_unpack_data::~teca_unpack_data()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_unpack_data::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_unpack_data":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, output_data_type,
            "Sets the type of the transformed data to either single or double"
            " precision floating point. Use 11 for single precision and 12 for"
            " double precision.")
        TECA_POPTS_GET(int, prefix, verbose, "Enables verbose output")
        ;


    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_unpack_data::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, output_data_type)
    TECA_POPTS_SET(opts, int, prefix, verbose)
}
#endif

// --------------------------------------------------------------------------
int teca_unpack_data::validate_output_data_type(int val)
{
    // validate the output type
    if ((val != ((int)teca_variant_array_code<float>::get())) &&
        (val != ((int)teca_variant_array_code<double>::get())))
    {
        TECA_ERROR("Invlaid output data type " << val << ". Use "
            << teca_variant_array_code<double>::get()
            << " to select double precision output and "
            << teca_variant_array_code<float>::get()
            << " to select single precision output")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_unpack_data::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_unpack_data::get_output_metadata" << endl;
#endif
    (void)port;

    // for each array on the input look for the presence of scale_factor and
    // add_offset if both attributes are present then modify the output data
    // type.
    teca_metadata out_md(input_md[0]);

    std::vector<std::string> variables;
    if (out_md.get("variables", variables))
    {
        TECA_FATAL_ERROR("Failed to get the list of variables")
        return teca_metadata();
    }

    teca_metadata attributes;
    if (out_md.get("attributes", attributes))
    {
        TECA_FATAL_ERROR("Failed to get the array attributes")
        return teca_metadata();
    }

    size_t n_vars = variables.size();
    for (size_t i = 0; i < n_vars; ++i)
    {
        const std::string &array_name = variables[i];

        teca_metadata array_atts;
        if (attributes.get(array_name, array_atts))
        {
            // this could be reported as an error or a warning but unless this
            // becomes problematic quietly ignore it
            continue;
        }

        // if both scale_factor and add_offset  attributes are present then
        // the data will be transformed. Update the output type.
        if (array_atts.has("scale_factor") && array_atts.has("add_offset"))
        {
            array_atts.set("type_code", this->output_data_type);

            array_atts.remove("scale_factor");
            array_atts.remove("add_offset");

            if (array_atts.has("_FillValue") || array_atts.has("missing_value"))
            {
               array_atts.remove("_FillValue");
               array_atts.remove("missing_value");

               if (this->output_data_type == ((int)teca_variant_array_code<double>::get()))
                   array_atts.set("_FillValue", 1e20);
               else if (this->output_data_type == ((int)teca_variant_array_code<float>::get()))
                   array_atts.set("_FillValue", 1e20f);
            }

            attributes.set(array_name, array_atts);
        }
    }

    out_md.set("attributes", attributes);
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_unpack_data::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    // get the list of variable available. we need to see if
    // the valid value mask is available and if so request it
    const teca_metadata &md = input_md[0];

    std::set<std::string> variables;
    if (md.get("variables", variables))
    {
        TECA_FATAL_ERROR("Metadata issue. variables is missing")
        return up_reqs;
    }

    teca_metadata attributes;
    if (md.get("attributes", attributes))
    {
        TECA_FATAL_ERROR("Failed to get the array attributes")
        return up_reqs;
    }

    // add the dependent variables into the requested arrays
    std::set<std::string> arrays_up;
    if (req.has("arrays"))
        req.get("arrays", arrays_up);

    std::vector<std::string> arrays_in(arrays_up.begin(), arrays_up.end());
    int n_arrays = arrays_in.size();
    for (int i = 0; i < n_arrays; ++i)
    {
        const std::string &array_name = arrays_in[i];

        teca_metadata array_atts;
        if (attributes.get(array_name, array_atts))
        {
            // this could be reported as an error or a warning but unless this
            // becomes problematic quietly ignore it
            continue;
        }

        // if both scale_factor and add_offset  attributes are present then
        // the data will be transformed. Update the output type.
        if (array_atts.has("scale_factor") && array_atts.has("add_offset") &&
            (array_atts.has("_FillValue") || array_atts.has("missing_value")))
        {
            // request the valid value mask if they are available.
            std::string mask_var = array_name + "_valid";
            if (variables.count(mask_var))
                arrays_up.insert(mask_var);
        }
    }

    // update the request
    req.set("arrays", arrays_up);

    // send it up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_unpack_data::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_unpack_data::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Input dataset is not a teca_mesh")
        return nullptr;
    }

    p_teca_mesh out_mesh =
        std::static_pointer_cast<teca_mesh>(in_mesh->new_instance());

    out_mesh->shallow_copy(std::const_pointer_cast<teca_mesh>(in_mesh));

    teca_metadata attributes;
    if (out_mesh->get_attributes(attributes))
    {
        TECA_FATAL_ERROR("Failed to get attributes")
        return nullptr;
    }

    // for each array
    p_teca_array_collection point_arrays = out_mesh->get_point_arrays();
    int n_arrays = point_arrays->size();
    for (int i = 0; i < n_arrays; ++i)
    {
        const std::string &array_name = point_arrays->get_name(i);

        // skip valid value masks
        size_t len = array_name.size();
        if ((len > 6) && (strcmp("_valid", array_name.c_str() + len - 6) == 0))
            continue;

        // check if this array is to be transformed
        teca_metadata array_atts;
        double scale = 0.0;
        double offset = 0.0;
        if (attributes.get(array_name, array_atts) ||
            array_atts.get("scale_factor", scale) ||
            array_atts.get("add_offset", offset))
            continue;

        // check for valid value mask
        std::string mask_name = array_name + "_valid";
        p_teca_variant_array mask = point_arrays->get(mask_name);

        // get the input
        p_teca_variant_array in_array = point_arrays->get(i);

        p_teca_variant_array out_array;
#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (teca_cuda_util::set_device(device_id))
            return nullptr;

        if (cuda_gpu::dispatch(device_id, in_array, mask, scale,
            offset, out_array, this->output_data_type))
        {
            TECA_ERROR("Failed to compute unpack data on the CPU")
            return nullptr;
        }
    }
    else
    {
#endif
        if (cpu::dispatch(in_array, mask, scale, offset,
            out_array, this->output_data_type))
        {
            TECA_ERROR("Failed to compute unpack data on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif
        // poass to the output
        point_arrays->set(i, out_array);

        // update the metadata
        array_atts.set("type_code", this->output_data_type);
        attributes.set(array_name, array_atts);

        if (this->verbose)
        {
            TECA_STATUS("Unpacked \"" << array_name << "\" scale_factor = "
                << scale << " add_offset = " << offset)
        }
    }

    return out_mesh;
}
