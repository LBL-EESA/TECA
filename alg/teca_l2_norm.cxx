#include "teca_l2_norm.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

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
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

namespace teca_l2_norm_internals
{
// CPU codes
namespace cpu
{
// ***************************************************************************
template <typename T>
void square(T *s, const T *c, unsigned long n_elem)
{
    for (unsigned long i = 0; i < n_elem; ++i)
    {
        T ci = c[i];
        s[i] = ci*ci;
    }
}

// ***************************************************************************
template <typename T>
void sum_square(T *ss, const T *c, unsigned long n_elem)
{
    for (unsigned long i = 0; i < n_elem; ++i)
    {
        T ci = c[i];
        ss[i] += ci*ci;
    }
}

// ***************************************************************************
template <typename T>
void square_root(T *rt, const T *c, unsigned long n_elem)
{
    for (unsigned long i = 0; i < n_elem; ++i)
    {
        rt[i] = std::sqrt(c[i]);
    }
}

// ***************************************************************************
int dispatch(p_teca_variant_array &l2_norm,
    const const_p_teca_variant_array &c0, const const_p_teca_variant_array &c1,
    const const_p_teca_variant_array &c2)
{
    // allocate the output array
    unsigned long n_elem = c0->size();

    l2_norm = c0->new_instance(n_elem);
    p_teca_variant_array tmp = c0->new_instance(n_elem);

    // compute l2 norm
    VARIANT_ARRAY_DISPATCH_FP(l2_norm.get(),

        auto [ptmp, pl2n] = data<TT>(tmp, l2_norm);
        auto [spc0, pc0] = get_host_accessible<CTT>(c0);

        sync_host_access_any(c0);
        cpu::square(ptmp, pc0, n_elem);

        if (c1)
        {
            auto [spc1, pc1] = get_host_accessible<CTT>(c1);
            sync_host_access_any(c1);
            cpu::sum_square(ptmp, pc1, n_elem);
        }

        if (c2)
        {
            auto [spc2, pc2] = get_host_accessible<CTT>(c2);
            sync_host_access_any(c2);
            cpu::sum_square(ptmp, pc2, n_elem);
        }

        cpu::square_root(pl2n, ptmp, n_elem);

        return 0;
        )


    TECA_ERROR("Unsupported type " << c0->get_class_name() << " for L2 norm")
    return -1;
}
}




#if defined(TECA_HAS_CUDA)
// CUDA codes
namespace cuda_gpu
{
// ***************************************************************************
template <typename T>
__global__
void l2_norm(T *nm, const T *c0, const T *c1, const T *c2, unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    T c0i = c0[i];
    T c1i = c1 ? c1[i] : T(0);
    T c2i = c2 ? c2[i] : T(0);

    T ss = c0i*c0i + c1i*c1i + c2i*c2i;

    nm[i] = sqrt(ss);
}

// ***************************************************************************
template <typename T>
int l2_norm(int device_id, T *pl2n, const T *pc0, const T *pc1,
    const T *pc2, unsigned long n_elem)
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

    // launch the l2 norm kernel
    cudaError_t ierr = cudaSuccess;
    l2_norm<<<block_grid,thread_grid>>>(pl2n, pc0, pc1, pc2, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the l2_norm CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}


// ***************************************************************************
int dispatch(int device_id, p_teca_variant_array &l2_norm,
    const const_p_teca_variant_array &c0, const const_p_teca_variant_array &c1,
    const const_p_teca_variant_array &c2)
{
    // set the CUDA device to run on
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to set the CUDA device to " << device_id
            << ". " << cudaGetErrorString(ierr))
        return -1;
    }

    // allocate the output array
    unsigned long n_elem = c0->size();
    if (n_elem < 1)
    {
        TECA_ERROR("Empty input array")
        return -1;
    }

    l2_norm = c0->new_instance(n_elem, allocator::cuda_async);

    // compute l2 norm
    VARIANT_ARRAY_DISPATCH_FP(l2_norm.get(),

        auto [pl2n] = data<TT>(l2_norm);
        auto [spc0, pc0] = get_cuda_accessible<CTT>(c0);

        CSP spc1;
        const NT *pc1 = nullptr;
        if (c1)
            std::tie(spc1, pc1) = get_cuda_accessible<CTT>(c1);

        CSP spc2;
        const NT *pc2 = nullptr;
        if (c2)
            std::tie(spc2, pc2) = get_cuda_accessible<CTT>(c2);

        if (cuda_gpu::l2_norm(device_id, pl2n, pc0, pc1, pc2, n_elem))
            return -1;

        return 0;
        )

    TECA_ERROR("Unsupported type " << c0->get_class_name() << " for L2 norm")
    return -1;
}
}
#endif
}


// --------------------------------------------------------------------------
teca_l2_norm::teca_l2_norm() :
    component_0_variable(), component_1_variable(),
    component_2_variable(), l2_norm_variable("l2_norm")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_l2_norm::~teca_l2_norm()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_l2_norm::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_l2_norm":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable,
            "array containg the first component")
        TECA_POPTS_GET(std::string, prefix, component_1_variable,
            "array containg the second component")
        TECA_POPTS_GET(std::string, prefix, component_2_variable,
            "array containg the third component")
        TECA_POPTS_GET(std::string, prefix, l2_norm_variable,
            "array to store the computed norm in")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_l2_norm::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, component_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_1_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_2_variable)
    TECA_POPTS_SET(opts, std::string, prefix, l2_norm_variable)
}
#endif

/*
// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_0_variable(const teca_metadata &request)
{
    std::string comp_0_var = this->component_0_variable;

    if (comp_0_var.empty() &&
        request.has("teca_l2_norm::component_0_variable"))
            request.get("teca_l2_norm::component_0_variable", comp_0_var);

    return comp_0_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_1_variable(
    const teca_metadata &request)
{
    std::string comp_1_var = this->component_1_variable;

    if (comp_1_var.empty() &&
        request.has("teca_l2_norm::component_1_variable"))
            request.get("teca_l2_norm::component_1_variable", comp_1_var);

    return comp_1_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_component_2_variable(const teca_metadata &request)
{
    std::string comp_2_var = this->component_2_variable;

    if (comp_2_var.empty() &&
        request.has("teca_l2_norm::component_2_variable"))
            request.get("teca_l2_norm::component_2_variable", comp_2_var);

    return comp_2_var;
}

// --------------------------------------------------------------------------
std::string teca_l2_norm::get_l2_norm_variable(const teca_metadata &request)
{
    std::string norm_var = this->l2_norm_variable;

    if (norm_var.empty())
    {
        if (request.has("teca_l2_norm::l2_norm_variable"))
            request.get("teca_l2_norm::l2_norm_variable", norm_var);
        else
            norm_var = "l2_norm";
    }

    return norm_var;
}
*/

// --------------------------------------------------------------------------
teca_metadata teca_l2_norm::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_l2_norm::get_output_metadata" << std::endl;
#endif
    (void)port;

    if (this->component_0_variable.empty())
    {
        TECA_FATAL_ERROR("The component_0_variable was not set")
        return teca_metadata();
    }

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    std::string norm_var = this->l2_norm_variable;
    if (norm_var.empty())
        norm_var = "l2_norm";

    out_md.append("variables", norm_var);

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    teca_metadata comp_0_atts;
    if (attributes.get(this->component_0_variable, comp_0_atts))
    {
        TECA_WARNING("Failed to get component 0 \"" << this->component_0_variable
            << "\" attrbibutes. Writing the result will not be possible")
    }
    else
    {
        // copy the attributes from the input. this will capture the
        // data type, size, units, etc.
        teca_array_attributes norm_atts(comp_0_atts);

        // update name, long_name, and description.
        norm_atts.long_name = norm_var;

        norm_atts.description =
            std::string("The L2 norm of (" + this->component_0_variable);
        if (!this->component_1_variable.empty())
            norm_atts.description += ", " + this->component_1_variable;
        if (!this->component_2_variable.empty())
            norm_atts.description += ", " + this->component_2_variable;
        norm_atts.description += ")";

        attributes.set(norm_var, (teca_metadata)norm_atts);
        out_md.set("attributes", attributes);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_l2_norm::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // check required variable names
    if (this->component_0_variable.empty())
    {
        TECA_FATAL_ERROR("component 0 variable was not specified")
        return up_reqs;
    }

    if (this->l2_norm_variable.empty())
    {
        TECA_FATAL_ERROR("L2 norm variable was not specified")
        return up_reqs;
    }

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->component_0_variable);

    if (!this->component_1_variable.empty())
        arrays.insert(this->component_1_variable);

    if (!this->component_2_variable.empty())
        arrays.insert(this->component_2_variable);

    // intercept request for our output
    arrays.erase(this->l2_norm_variable);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_l2_norm::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_l2_norm::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to compute l2 norm. dataset is not a teca_mesh")
        return nullptr;
    }

    // get the name of the first component
    if (this->component_0_variable.empty())
    {
        TECA_FATAL_ERROR("component 0 array was not specified")
        return nullptr;
    }

    // get the first component array
    const_p_teca_variant_array c0
        = in_mesh->get_point_arrays()->get(this->component_0_variable);
    if (!c0)
    {
        TECA_FATAL_ERROR("component 0 array \"" << this->component_0_variable
            << "\" not present.")
        return nullptr;
    }

    // get the optional component arrays
    const_p_teca_variant_array c1 = nullptr;
    if (!this->component_1_variable.empty() &&
        !(c1 = in_mesh->get_point_arrays()->get(this->component_1_variable)))
    {
        TECA_FATAL_ERROR("component 1 array \"" << this->component_1_variable
            << "\" requested but not present.")
        return nullptr;
    }

    const_p_teca_variant_array c2 = nullptr;
    if (!this->component_2_variable.empty() &&
        !(c2 = in_mesh->get_point_arrays()->get(this->component_2_variable)))
    {
        TECA_FATAL_ERROR("component 2 array \"" << this->component_2_variable
            << "\" requested but not present.")
        return nullptr;
    }

    p_teca_variant_array l2_norm;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (teca_l2_norm_internals::cuda_gpu::dispatch(device_id, l2_norm, c0, c1, c2))
        {
            TECA_ERROR("Failed to compute the L2 norm using CUDA")
            return nullptr;
        }
    }
    else
    {
#endif
        if (teca_l2_norm_internals::cpu::dispatch(l2_norm, c0, c1, c2))
        {
            TECA_ERROR("Failed to compute the L2 norm on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // create the output mesh, pass everything through, and
    // add the l2 norm array
    p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());

    out_mesh->get_point_arrays()->set(this->l2_norm_variable, l2_norm);

    return out_mesh;
}
