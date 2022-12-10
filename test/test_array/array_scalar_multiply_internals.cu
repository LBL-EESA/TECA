#include "array_scalar_multiply_internals.h"

#include "teca_cuda_util.h"

namespace array_scalar_multiply_internals
{
namespace gpu
{
// **************************************************************************
template<typename data_t>
__global__
void multiply(data_t *result, const data_t *array_in,
    data_t scalar, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_in[i] * scalar;
}
}

// **************************************************************************
int cuda_dispatch(int device_id, p_array &result,
    const const_p_array &array_in, double scalar, size_t n_vals)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_scalar_multiply_internals::cuda_dispatch device_id="
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

    // make sure that inputs are on the GPU
    std::shared_ptr<const double> parray_in = array_in->get_cuda_accessible();

    // allocate the memory
    result = array::new_cuda_accessible();
    result->resize(n_vals);

    // determine kernel launch parameters
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // add the arrays
    array_scalar_multiply_internals::gpu::multiply<<<block_grid, thread_grid>>>(
        result->data(), parray_in.get(), scalar, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the multiply kernel. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // sync
    cudaDeviceSynchronize();

    return 0;
}
}
