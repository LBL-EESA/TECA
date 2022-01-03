#ifndef teca_cuda_util_h
#define teca_cuda_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_mpi.h"

#include <deque>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>


/// A collection of utility classes and functions for integrating with CUDA
namespace teca_cuda_util
{

/** query the system for the locally available(on this rank) CUDA device count.
 * this is an MPI collective call which returns a set of device ids that can be
 * used locally. If there are as many (or more than) devices on the node than
 * the number of MPI ranks assigned to the node the list of device ids will be
 * unique across MPI ranks on the node. Otherwise devices are assigned round
 * robin fashion.
 *
 * @param[in]  comm      MPI communicator defining a set of nodes on which need
 *                       access to the available GPUS
 * @param[out] local_dev a list of device ids that can be used by the calling
 *                       MPI rank.
 * @returns              non-zero on error.
 */
TECA_EXPORT
int get_local_cuda_devices(MPI_Comm comm, std::vector<int> &local_dev);

/// set the CUDA device. returns non-zero on error
TECA_EXPORT
int set_device(int device_id);

/// stop and wait for previously launched kernels to complete
TECA_EXPORT
int synchronize();

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

/** Calculate CUDA launch parameters for an arbitrarily large flat array.
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

/** calculate CUDA launch parameters for an arbitrarily large flat array. See
 * ::get_launch_props for how to query CUDA for block_grid_max and warp_size
 * parameters.
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
}

///@}
#endif
