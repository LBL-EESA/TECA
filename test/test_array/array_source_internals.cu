#include "array_source_internals.h"
#include "teca_cuda_util.h"

namespace array_source_internals
{
namespace gpu
{
// **************************************************************************
template<typename data_t>
__global__
void initialize(data_t *data, double val, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    data[i] = val;
}
}

// **************************************************************************
int cuda_dispatch(int device_id, p_array &a_out, double val, size_t n_vals)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_source_internals::cuda_dispatch device_id="
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

    // allocate the memory
    a_out = array::new_cuda_accessible();
    a_out->resize(n_vals);

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

    // initialize the data
    array_source_internals::gpu::initialize
        <<<block_grid, thread_grid>>>(a_out->data(), val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the initialize kernel. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

}
