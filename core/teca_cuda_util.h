#ifndef teca_cuda_util_h
#define teca_cuda_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_mpi.h"

#include <deque>
#include <vector>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>


/// A collection of utility classes and functions for integrating with CUDA
namespace teca_cuda_util
{
/** a wrapper for using dynamic sized shared memory in a template function.
 */
template <typename T>
__device__ T* shared_memory_proxy()
{
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

/** Query the system for the locally available(on this rank) CUDA device count.
 * this is an MPI collective call which returns a set of device ids that can be
 * used locally. Node wide coordination assures that one can put a limit on the
 * number of ranks per node.
 *
 * @param[in]  comm MPI communicator defining a set of nodes on which need
 *                  access to the available GPUS
 * @param[in,out] ranks_per_device The number of MPI ranks to use per CUDA
 *                                 device. When set to 0 no GPUs are used. When
 *                                 set to -1 all ranks are assigned a GPU but
 *                                 multiple ranks will share a GPU when there
 *                                 are more ranks than devices.
 *
 * @param[out] local_dev a list of device ids that can be used by the calling
 *                       MPI rank.
 * @returns              non-zero on error.
 */
TECA_EXPORT
int get_local_cuda_devices(MPI_Comm comm, int &ranks_per_device,
    std::vector<int> &local_dev);

/// set the CUDA device. returns non-zero on error
TECA_EXPORT
int set_device(int device_id);

/// device wide synchronize
TECA_EXPORT
int synchronize_device();

/// synchronize the default stream
TECA_EXPORT
int synchronize_stream();

/** A flat array is broken into blocks of number of threads where each adjacent
 * thread accesses adjacent memory locations. To accomplish this we might need
 * a large number of blocks. If the number of blocks exceeds the max block
 * dimension in the first and or second block grid dimension then we need to
 * use a 2d or 3d block grid.
 *
 * ::partition_thread_blocks - decides on a partitioning of the data based on
 * warps_per_block parameter. The resulting decomposition will be either 1,2,
 * or 3D as needed to accommodate the number of fixed sized blocks. It can
 * happen that max grid dimensions are hit, in which case you'll need to
 * increase the number of warps per block.
 *
 * ::thread_id_to_array_index - given a thread and block id gets the
 * array index to update. _this may be out of bounds so be sure
 * to validate before using it.
 *
 * ::index_is_valid - test an index for validity.
*/
/// @name CUDA indexing scheme
///@{

/** convert a CUDA index into a flat array index using the partitioning scheme
 * defined in ::partition_thread_blocks
 */
inline
__device__
unsigned long thread_id_to_array_index()
{
    return threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y);
}

/** convert a CUDA index into a flat array index using the partitioning scheme
 * defined in ::partition_thread_blocks_slab. This gives the index of the first
 * element in the vertical column.
 */
inline
__device__
void thread_id_to_array_index_slab(unsigned long &i, unsigned long &k0,
    unsigned long stride)
{
    // index in the xy slab
    i = threadIdx.x + blockDim.x * blockIdx.x;

    // first index in the vertical dimension
    k0 = stride * blockIdx.y;
}

/// bounds check the flat index
inline
__device__
int index_is_valid(unsigned long index, unsigned long max_index)
{
    return index < max_index;
}

/** query properties for the named CUDA device.
 * @param[in]  device_id The device to query, or -1 for the active device.
 * @param[out] block_grid_max a 3 value array of the maximum number of thread
 *                            blocks supported in x,y, and z directions.
 * @param[out] warp_size the number of threads per warp
 * @param[out] max_warps_per_block the maximum number of warps per block
 *                                 supported.
 * @returns non-zero on error.
 */
TECA_EXPORT
int get_launch_props(int device_id,
    int *block_grid_max, int &warp_size,
    int &max_warps_per_block);

/** Calculate CUDA launch parameters for an arbitrarily large flat array. The
 * block grid will be 1d if the device can process the array using a 1d block
 * grid, otherwise dimensions are added to accomodate the array size up to the
 * largest grid supported by the device. Use ::thread_id_to_array_index in the
 * kernel to determine the array index to process.
 *
 * @param[in]  device_id       the CUDA device to query launch parameter limits from.
 *                             Use -1 to query from the currently active device.
 * @param[in]  array_size      the length of the array being processed
 * @param[in]  warps_per_block the number of warps to use per block (your choice)
 * @param[out] block_grid      the block dimension kernel launch control
 * @param[out] n_blocks        the number of blocks
 * @param[out] thread_grid     the thread dimension kernel launch control
 *
 * @returns zero if successful and non-zero if an error occurred
 */
TECA_EXPORT
int partition_thread_blocks(int device_id, size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks,
    dim3 &thread_grid);

/** Calculate CUDA launch parameters for an arbitrarily large flat array. The
 * block grid will be 1d if the device can process the array using a 1d block
 * grid, otherwise dimensions are added to accomodate the array size up to the
 * largest grid supported by the device. Use ::thread_id_to_array_index in the
 * kernel to determine the array index to process.  See ::get_launch_props for
 * how to query CUDA for block_grid_max and warp_size parameters.
 *
 * @param[in]  array_size      the length of the array being processed
 * @param[in]  warps_per_block the number of warps to use per block (your choice)
 * @param[in]  warp_size       the number of threads per warp
 * @param[in]  block_grid_max  the maximum number of blocks in the 3D block
 *                             grid supported by the CUDA device
 * @param[out] block_grid      the block dimension kernel launch control parameter
 * @param[out] n_blocks        the number of blocks
 * @param[out] thread_grid     the thread dimension kernel launch control parameter
 *
 * @returns zero if successful and non-zero if an error occurred
 */
TECA_EXPORT
int partition_thread_blocks(size_t array_size,
    int warps_per_block, int warp_size, int *block_grid_max,
    dim3 &block_grid, int &n_blocks, dim3 &thread_grid);

/** Calculate CUDA launch parameters for an arbitrarily large 3D array that
 * will be processed by looping over vertical the vertical dimension.
 * Partitioning in the first dimension occurs over x-y slab sized sections of
 * the array. In the second dimension the caller declares the number of
 * elements (stride) desired in vertical dimension loops.  A 2d block grid will
 * be generated up to the limits of the selected device. Use
 * ::thread_id_to_array_index_slab in the kernel to determine the array index
 * to process.
 *
 * @param[in]  device_id       the CUDA device to query launch parameter limits from.
 *                             Use -1 to query from the currently active device.
 * @param[in]  nxy             the size in the xy dimension of the array being processed
 * @param[in]  nz              the size in the vertical dimension of the array being processed
 * @param[in]  warps_per_block the number of warps to use per block (your choice)
 * @param[out] block_grid      the block dimension kernel launch control
 * @param[out] n_blocks_xy     the number of nxy sized blocks
 * @param[out] n_blocks_z      the number of blocks in the vertical dimension
 * @param[out] thread_grid     the thread dimension kernel launch control
 *
 * @returns zero if successful and non-zero if an error occurred
 */
TECA_EXPORT
int partition_thread_blocks_slab(int device_id, size_t nxy, size_t nz,
    size_t stride, int warps_per_block, dim3 &block_grid, int &n_blocks_xy,
    int &n_blocks_z, dim3 &thread_grid);

/** Calculate CUDA launch parameters for an arbitrarily large 3D array that
 * will be processed by looping over vertical the vertical dimension.
 * Partitioning in the first dimension occurs over x-y slab sized sections of
 * the array. In the second dimension the caller declares the number of
 * elements (stride) desired in vertical dimension loops. Kernels will compute
 * the loop bounds from the stride and block index.  A 2d block grid will be
 * generated up to the limits of the selected device. Use
 * ::thread_id_to_array_index_slab in the kernel to determine the array index
 * to process.  See ::get_launch_props for how to query CUDA for block_grid_max
 * and warp_size parameters.
 *
 * @param[in]  nxy             the length of the array being processed
 * @param[in]  warps_per_block the number of warps to use per block (your choice)
 * @param[in]  warp_size       the number of threads per warp
 * @param[in]  block_grid_max  the maximum number of blocks in the 3D block
 *                             grid supported by the CUDA device
 * @param[out] block_grid      the block dimension kernel launch control parameter
 * @param[out] n_blocks_xy     the number of nxy sized blocks
 * @param[out] n_blocks_z      the number of blocks in the vertical direction
 * @param[out] thread_grid     the thread dimension kernel launch control parameter
 *
 * @returns zero if successful and non-zero if an error occurred
 */
TECA_EXPORT
int partition_thread_blocks_slab(size_t nxy, size_t nz, size_t stride,
    int warps_per_block, int warp_size, int *block_grid_max, dim3 &block_grid,
    int &n_blocks_xy, int &n_blocks_z,  dim3 &thread_grid);

/** Calculate CUDA launch parameters for an arbitrarily large 1D array.
 * @param[in] nt the number of threads per block
 * @param[in] n_vals the size of the 1D array
 * @returns a tuple containing the number of thread blocks and the number of
 *          threads per block
 */
inline
auto partition_thread_blocks_1d(unsigned int nt, size_t n_vals)
{
    return std::make_tuple((n_vals / nt + (n_vals % nt ? 1 : 0)), nt);
}
///@}



/// A collection of CUDA streams.
/** This container always has as its first element the cudaStreamPerThread
 * stream. If more than one stream is desired one can call ::resize to add
 * new streams to the collection. For simplicity copying the container is
 * disabled, but this feature could be added if needed.
 */
class TECA_EXPORT cuda_stream_vector
{
public:
    cuda_stream_vector() : m_vec(1, cudaStreamPerThread) {}
    ~cuda_stream_vector();

    /// prevent copies, OK to enable these if needed
    cuda_stream_vector(const cuda_stream_vector &) = delete;
    void operator=(const cuda_stream_vector &) = delete;

    /// resize the collection. creates and destroys streams as needed
    int resize(size_t n);

    /// get the number of available cuda streams
    size_t size() const { return m_vec.size(); }

    /// get the ith cuda stream
    cudaStream_t &operator[](size_t i) { return m_vec[i]; }

    /// get the ith cuda stream
    const cudaStream_t &operator[](size_t i) const { return m_vec[i]; }

private:
    std::vector<cudaStream_t> m_vec;
};

}
#endif
