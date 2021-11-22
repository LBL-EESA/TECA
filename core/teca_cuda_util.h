#ifndef teca_cuda_util_h
#define teca_cuda_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_mpi.h"

#include <deque>

#include <cuda.h>
#include <cuda_runtime.h>


/// A collection of utility classes and functions for intergacing with CUDA
namespace teca_cuda_util
{

/** query the system for the locally available(on this rank) CUDA device count.
 * this is an MPI collective call which returns a set of device ids that can be
 * used locally. If there are as many (or more than) devices on the node than
 * the number of MPI ranks assigned to the node the list of devicce ids will be
 * unique across MPI ranks on the node. Otherwise devices are assigned round
 * robbin fashion.
 *
 * @param[in]  comm      MPI communicator defining a set of nodes on which need
 *                       access to the available GPUS
 * @param[out] local_dev a list of device ids that can be used my the calling
 *                       MPI rank.
 * @returns              non-zero on error.
 */
TECA_EXPORT
int get_local_cuda_devices(MPI_Comm comm, std::deque<int> &local_dev);

/// set the CUDA device. returns non-zero on error
TECA_EXPORT
int set_device(int device_id);

/// stop and wait for previuoiusly launched kernels to complete
TECA_EXPORT
int synchronize();

/// querry properties for the named CUDA device. retruns non-zero on error
TECA_EXPORT
int get_launch_props(int device_id,
    int *block_grid_max, int &warp_size,
    int &max_warps_per_block);

/** A flat array is broken into blocks of number of threads where each adjacent
 * thread accesses adjacent memory locations. To accomplish this we might need
 * a large number of blocks. If the number of blocks exceeds the max block
 * dimension in the first and or second block grid dimension then we need to
 * use a 2d or 3d block grid.
 *
 * partition_thread_blocks - decides on a partitioning of the data based on
 * warps_per_block parameter. The resulting decomposition will be either 1,2,
 * or 3D as needed to accomodate the number of fixed sized blocks. It can
 * happen that max grid dimensions are hit, in which case you'll need to
 * increase the number of warps per block.
 *
 * thread_id_to_array_index - given a thread and block id gets the
 * array index to update. _this may be out of bounds so be sure
 * to validate before using it.
 *
 * index_is_valid - test an index for validity.
*/
/// @name CUDA indexing scheme
///@{

/** convert a CUDA index into a flat array index using the paritioning scheme
 * defined in partition_thread_blocks
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

// calculate CUDA launch paramters for an arbitrarily large flat array
//
// inputs:
//      array_size -- the length of the array being processed
//      warps_per_block -- number of warps to use per block (your choice)
//
// outputs:
//      block_grid -- block dimension kernel launch control
//      n_blocks -- number of blocks
//      thread_grid -- thread dimension kernel launch control
//
// returns:
//      non zero on error
TECA_EXPORT
int partition_thread_blocks(int device_id, size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks,
    dim3 &thread_grid);

// calculate CUDA launch paramters for an arbitrarily large flat array
//
// inputs:
//      array_size -- the length of the array being processed
//      warp_size -- number of threads per warp supported on the device
//      warps_per_block -- number of warps to use per block (your choice)
//      block_grid_max -- maximum number of blocks supported by the device
//
// outputs:
//      block_grid -- block dimension kernel launch control
//      n_blocks -- number of blocks
//      thread_grid -- thread dimension kernel launch control
//
// returns:
//      non zero on error
TECA_EXPORT
int partition_thread_blocks(size_t array_size,
    int warps_per_block, int warp_size, int *block_grid_max,
    dim3 &block_grid, int &n_blocks, dim3 &thread_grid);
}

///@}
#endif
