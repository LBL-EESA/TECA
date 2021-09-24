#include "array_temporal_stats_internals.h"
#include "array_util.h"

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

namespace array_temporal_stats_internals
{
namespace gpu
{
// **************************************************************************
template<typename data_t>
void compute_stats(data_t *array_out, const data_t *array_in, size_t n_vals)
{
    // important! array_out is always on the CPU
    thrust::device_ptr<const data_t> ptr = thrust::device_pointer_cast<const data_t>(array_in);

    // min
    array_out[0] = thrust::reduce(ptr, ptr + n_vals,
        std::numeric_limits<data_t>::max(), thrust::minimum<data_t>());

    // max
    array_out[1] = thrust::reduce(ptr, ptr + n_vals,
        std::numeric_limits<data_t>::lowest(), thrust::maximum<data_t>());

    // sum
    array_out[2] = thrust::reduce(ptr, ptr + n_vals,
        data_t(0), thrust::plus<data_t>());

    // count
    array_out[3] = n_vals;
}
}

// **************************************************************************
int cuda_dispatch(int device_id, p_array &results, const const_p_array &l_in,
    const const_p_array &r_in, bool l_active, bool r_active)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_temporal_stats_internals::cuda_dispatch device_id="
        << device_id << std::endl;
#endif

    // set the device
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to set the CUDA device to " << device_id
            << ". " << cudaGetErrorString(ierr))
        return -1;
    }

    // allocate space for the output
    results = array::new_cpu_accessible();
    results->resize(4);

    // cases:
    if (l_active && r_active)
    {
        // both left and right contain new data
        // compute stats from left
        const_p_array tmp_l_in = array_util::cuda_accessible(l_in);

        p_array tmp_res_l = array::new_cpu_accessible();
        tmp_res_l->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            tmp_res_l->get(), tmp_l_in->get(), tmp_l_in->size());

        // compute stats from right
        const_p_array tmp_r_in = array_util::cuda_accessible(r_in);

        p_array tmp_res_r = array::new_cpu_accessible();
        tmp_res_r->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            tmp_res_r->get(), tmp_r_in->get(), tmp_r_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), tmp_res_l->get(), tmp_res_r->get());
    }
    else
    if (l_active)
    {
        // left contains new data, right contains result

        // compute stats from left
        const_p_array tmp_l_in = array_util::cuda_accessible(l_in);

        p_array tmp_res_l = array::new_cpu_accessible();
        tmp_res_l->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            tmp_res_l->get(), tmp_l_in->get(), tmp_l_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), tmp_res_l->get(), r_in->get());
    }
    else
    if (r_active)
    {
        // right contains data, left contains result

        // compute stats from right
        const_p_array tmp_r_in = array_util::cuda_accessible(r_in);

        p_array tmp_res_r = array::new_cpu_accessible();
        tmp_res_r->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            tmp_res_r->get(), tmp_r_in->get(), tmp_r_in->size());

        // reduce stats from the left (always on CPU)
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), l_in->get(), tmp_res_r->get());
    }
    else
    {
        // both left and right contain stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), l_in->get(), r_in->get());
    }

    return 0;
}
}
