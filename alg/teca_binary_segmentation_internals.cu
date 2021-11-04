#include "teca_binary_segmentation_internals.h"
#include "teca_variant_array.h"
#include "teca_cuda_util.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace teca_binary_segmentation_internals
{
namespace gpu
{
// predicate for indirect sort
template <typename data_t, typename index_t>
struct indirect_lt
{
    indirect_lt() : p_data(nullptr) {}
    indirect_lt(const data_t *pd) : p_data(pd) {}

    __device__
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

    __device__
    bool operator()(const index_t &a, const index_t &b)
    {
        return p_data[a] > p_data[b];
    }

    const data_t *p_data;
};

// set locations in the output where the input array
// has values within the low high range.
template <typename input_t, typename output_t>
__global__
void value_threshold_kernel(output_t *output, const input_t *input,
    size_t n_vals, input_t low, input_t high)
{
    // get a tuple from the current flat index in the output
    // index space
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

// set locations in the output where the input array
// has values within the low high range.
template <typename input_t, typename output_t>
int value_threshold(output_t *dev_output, const input_t *dev_input,
    size_t n_vals, input_t low, input_t high)
{
/*
    cudaError_t ierr;
    // allocate and send the input to the device
    input_t *dev_input = nullptr;
    if (((ierr = cudaMalloc(&dev_input, n_vals*sizeof(input_t)))) ||
        ((ierr = cudaMemcpy(dev_input, input, n_vals*sizeof(input_t),
        cudaMemcpyHostToDevice))))
    {
        TECA_ERROR("Failed to move input to the device. "
            <<  cudaGetErrorString(ierr))
        return -1;
    }

    // allocate a buffer for the segmentation
    output_t *dev_output = nullptr;
    if ((ierr = cudaMalloc(&dev_output, n_vals*sizeof(output_t))))
    {
        TECA_ERROR("Failed to allocate buffer for the output. "
            <<  cudaGetErrorString(ierr))
        return -1;
    }
*/
    // determine kernel launch parameters
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")

    }

    // segment using the values
    value_threshold_kernel<<<block_grid, thread_grid>>>(dev_output,
        dev_input, n_vals, low, high);
/*
    // copy the result back
    if ((ierr = cudaMemcpy(output, dev_output, n_vals*sizeof(output_t),
        cudaMemcpyDeviceToHost)))
    {
        TECA_ERROR("Failed to move output to host. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // free buffers
    cudaFree(dev_input);
    cudaFree(dev_output);
*/
    return 0;
}

// segments the input such that elements >= low_t and <= high_t
// are 1. the ids array contains the ids from an indirect sort
// such that low_cut, low_cut_p1, high_cut, high_cut_p1 hold
// the values needed to compute the desired percentile
template <typename index_t, typename input_t, typename output_t>
__global__
void percentile_threshold(output_t *output,
    const input_t *input, index_t *ids,
    index_t n_vals, index_t low_cut, index_t low_cut_p1,
    index_t high_cut, index_t high_cut_p1,
    double t_low, double t_high)
{
    // find 2 indices needed for low percentile calc
    double y0 = input[ids[low_cut]];
    double y1 = input[ids[low_cut_p1]];

    // compute low percetile
    double low_percentile = (y1 - y0)*t_low + y0;

    // find 2 indices needed for the high percentile calc
    y0 = input[ids[high_cut]];
    y1 = input[ids[high_cut_p1]];

    // compute high percentile
    double high_percentile = (y1 - y0)*t_high + y0;

    /*std::cerr << q_low << "th percentile is " <<  std::setprecision(10) << low_percentile << std::endl
        << q_high << "th percentile is " <<  std::setprecision(9) << high_percentile << std::endl;*/

    // get a tuple from the current flat index in the output
    // index space
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    output[i] = ((input[i] >= low_percentile) && (input[i] <= high_percentile)) ? 1 : 0;
}


// Given a vector V of length N, the q-th percentile of V is the value q/100 of
// the way from the minimum to the maximum in a sorted copy of V.

// set locations in the output where the input array
// has values within the low high range.
template <typename input_t, typename output_t>
int percentile_threshold(output_t *dev_output, const input_t *dev_input,
    unsigned long n_vals, float q_low, float q_high)
{
    using index_t = unsigned long;

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

    /*
    cudaError_t ierr;
    // allocate and send the input to the device
    input_t *dev_input = nullptr;
    if (((ierr = cudaMalloc(&dev_input, n_vals*sizeof(input_t)))) ||
        ((ierr = cudaMemcpy(dev_input, input, n_vals*sizeof(input_t),
        cudaMemcpyHostToDevice))))
    {
        TECA_ERROR("Failed to move input to the device. "
            <<  cudaGetErrorString(ierr))
        return -1;
    }

    // allocate a buffer for the segmentation
    output_t *dev_output = nullptr;
    if ((ierr = cudaMalloc(&dev_output, n_vals*sizeof(output_t))))
    {
        TECA_ERROR("Failed to allocate buffer for the output. "
            <<  cudaGetErrorString(ierr))
        return -1;
    }
    */

    // allocate indices and initialize them for the indirect sort
    thrust::device_vector<index_t> ids(n_vals);
    thrust::sequence(ids.begin(), ids.end());

    // sort the input field
    // use an indirect comparison that leaves the input data unmodified
    indirect_lt<input_t,index_t>  comp(dev_input);
    thrust::sort(ids.begin(), ids.end(), comp);

    index_t *dev_ids = thrust::raw_pointer_cast(ids.data());

    // determine kernel launch parameters
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")

    }

    // segment using the percentiles
    percentile_threshold<<<block_grid, thread_grid>>>(dev_output, dev_input,
        dev_ids, n_vals, low_cut, low_cut_p1, high_cut, high_cut_p1, t_low,
        t_high);

    /*
    // copy the result back
    if ((ierr = cudaMemcpy(output, dev_output, n_vals*sizeof(output_t),
        cudaMemcpyDeviceToHost)))
    {
        TECA_ERROR("Failed to move output to host. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // free buffers
    cudaFree(dev_input);
    cudaFree(dev_output);
    */

    return 0;
}
}

// **************************************************************************
int cuda_dispatch(int device_id,
    p_teca_variant_array &output_array,
    const const_p_teca_variant_array &input_array,
    int threshold_mode,
    double low, double high)
{
    // do segmentation
    size_t n_elem = input_array->size();

    p_teca_char_array segmentation =
        teca_char_array::New(n_elem, teca_variant_array::allocator::cuda);

    auto sp_seg = segmentation->get_cuda_accessible();
    char *p_seg = sp_seg.get();

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        input_array.get(),

        auto sp_in = static_cast<TT*>(input_array.get())->get_cuda_accessible();
        const NT *p_in = sp_in.get();

        if (threshold_mode == teca_binary_segmentation::BY_VALUE)
        {
            gpu::value_threshold(p_seg, p_in, n_elem,
               static_cast<NT>(low), static_cast<NT>(high));
        }
        else if  (threshold_mode == teca_binary_segmentation::BY_PERCENTILE)
        {
            gpu::percentile_threshold(p_seg, p_in, n_elem,
                static_cast<NT>(low), static_cast<NT>(high));
        }
        else
        {
            TECA_ERROR("Invalid threshold mode")
            return -1;
        }
        )

    //segmentation->debug_print();

/*    p_teca_int_array  tmp  = teca_int_array::New();
    tmp->assign(p_teca_variant_array(segmentation));
    tmp->print_buffer<int>();
*/
    output_array = segmentation;
    return 0;
}
}
