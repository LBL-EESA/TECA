#include "array_temporal_stats_internals.h"

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

    // NOTE : results are always on the CPU!

    // allocate space for the output
    results = array::new_host_accessible();
    results->resize(4);

    // cases:
    if (l_active && r_active)
    {
        // both left and right contain new data
        // compute stats from left
        std::shared_ptr<const double> pl_in = l_in->get_cuda_accessible();

        p_array res_l = array::new_host_accessible();
        res_l->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            res_l->data(), pl_in.get(), l_in->size());

        // compute stats from right
        std::shared_ptr<const double> pr_in = r_in->get_cuda_accessible();

        p_array res_r = array::new_host_accessible();
        res_r->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            res_r->data(), pr_in.get(), r_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), res_l->data(), res_r->data());
    }
    else
    if (l_active)
    {
        // left contains new data, right contains result

        // compute stats from left
        std::shared_ptr<const double> pl_in = l_in->get_cuda_accessible();

        p_array res_l = array::new_host_accessible();
        res_l->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            res_l->data(), pl_in.get(), l_in->size());

        // existing stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), res_l->data(), pr_in.get());
    }
    else
    if (r_active)
    {
        // right contains data, left contains result

        // compute stats from right
        std::shared_ptr<const double> pr_in = r_in->get_cuda_accessible();

        p_array res_r = array::new_host_accessible();
        res_r->resize(4);

        array_temporal_stats_internals::gpu::compute_stats(
            res_r->data(), pr_in.get(), r_in->size());

        // existing stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), pl_in.get(), res_r->data());
    }
    else
    {
        // both left and right contain stats

        // existing stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        // existing stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), pl_in.get(), pr_in.get());
    }

    return 0;
}
}
