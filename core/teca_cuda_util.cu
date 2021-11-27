#include "teca_cuda_util.h"

namespace teca_cuda_util
{
// **************************************************************************
int synchronize()
{
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaDeviceSynchronize()) != cudaSuccess)
    {
        TECA_ERROR("Failed to synchronize CUDA execution. "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// **************************************************************************
int get_local_cuda_devices(MPI_Comm comm, std::deque<int> &local_dev)
{
    cudaError_t ierr = cudaSuccess;

    // get the number of CUDA GPU's available on this node
    int n_node_dev = 0;
    if ((ierr = cudaGetDeviceCount(&n_node_dev)) != cudaSuccess)
    {
        TECA_ERROR("Failed to get the number of CUDA devices. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // if there are no GPU's error out
    if (n_node_dev < 1)
    {
        TECA_ERROR("No CUDA devices found")
        return -1;
    }

    // get the number of MPI ranks on this node, and their core id's
#if defined(TECA_HAS_MPI)
    int n_node_ranks = 1;
    int node_rank = 0;

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        // get node local rank and num ranks
        MPI_Comm node_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,
            0, MPI_INFO_NULL, &node_comm);

        MPI_Comm_size(node_comm, &n_node_ranks);
        MPI_Comm_rank(node_comm, &node_rank);

        if (n_node_dev >= n_node_ranks)
        {
            // assign devices evenly between ranks
            int max_dev = n_node_dev - 1;
            int n_per_rank = std::max(n_node_dev / n_node_ranks, 1);
            int n_larger = n_node_dev % n_node_ranks;

            int first_dev = n_per_rank * node_rank + (node_rank < n_larger ? node_rank : n_larger);
            first_dev = std::min(max_dev, first_dev);

            int last_dev = first_dev + n_per_rank - 1 + (node_rank < n_larger ? 1 : 0);
            last_dev = std::min(max_dev, last_dev);

            for (int i = first_dev; i <= last_dev; ++i)
                local_dev.push_back(i);
        }
        else
        {
            // round robbin assignment
            local_dev.push_back( node_rank % n_node_dev );
        }

        MPI_Comm_free(&node_comm);

        return 0;
    }
#endif
    // without MPI this process can use all CUDA devices
    for (int i = 0; i < n_node_dev; ++i)
        local_dev.push_back(i);
    return 0;
}


//-----------------------------------------------------------------------------
int set_device(int device_id)
{
    int n_devices = 0;

    cudaError_t ierr = cudaGetDeviceCount(&n_devices);
    if (ierr != cudaSuccess)
    {
        TECA_ERROR("Failed to get CUDA device count. "
            << cudaGetErrorString(ierr))
        return -1;
    }


    if (device_id >= n_devices)
    {
        TECA_ERROR("Attempt to select invalid device "
            << device_id << " of " << n_devices)
        return -1;
    }

    ierr = cudaSetDevice(device_id);
    if (ierr)
    {
        TECA_ERROR("Failed to select device " << device_id << ". "
            <<  cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int get_launch_props(int device_id,
    int *block_grid_max, int &warp_size,
    int &warps_per_block_max)
{
    cudaError_t ierr = cudaSuccess;

    if (((ierr = cudaDeviceGetAttribute(&block_grid_max[0], cudaDevAttrMaxGridDimX, device_id)) != cudaSuccess)
        || ((ierr = cudaDeviceGetAttribute(&block_grid_max[1], cudaDevAttrMaxGridDimY, device_id)) != cudaSuccess)
        || ((ierr = cudaDeviceGetAttribute(&block_grid_max[2], cudaDevAttrMaxGridDimZ, device_id)) != cudaSuccess))
    {
        TECA_ERROR("Failed to get CUDA max grid dim. " << cudaGetErrorString(ierr))
        return -1;
    }

    if ((ierr = cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to get CUDA warp size. " << cudaGetErrorString(ierr))
        return -1;
    }

    int threads_per_block_max = 0;

    if ((ierr = cudaDeviceGetAttribute(&threads_per_block_max,
        cudaDevAttrMaxThreadsPerBlock, device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to get CUDA max threads per block. " << cudaGetErrorString(ierr))
        return -1;
    }

    warps_per_block_max = threads_per_block_max / warp_size;

    return 0;
}

// --------------------------------------------------------------------------
int partition_thread_blocks(size_t array_size,
    int warps_per_block, int warp_size, int *block_grid_max,
    dim3 &block_grid, int &n_blocks, dim3 &thread_grid)
{
    unsigned long threads_per_block = warps_per_block * warp_size;

    thread_grid.x = threads_per_block;
    thread_grid.y = 1;
    thread_grid.z = 1;

    unsigned long block_size = threads_per_block;
    n_blocks = array_size / block_size;

    if (array_size % block_size)
        ++n_blocks;

    if (n_blocks > block_grid_max[0])
    {
        // multi-d decomp required
        block_grid.x = block_grid_max[0];
        block_grid.y = n_blocks / block_grid_max[0];
        if (n_blocks % block_grid_max[0])
        {
            ++block_grid.y;
        }

        if (block_grid.y > ((unsigned int)block_grid_max[1]))
        {
            // 3d decomp
            unsigned long block_grid_max01 = block_grid_max[0] * block_grid_max[1];
            block_grid.y = block_grid_max[1];
            block_grid.z = n_blocks / block_grid_max01;

            if (n_blocks % block_grid_max01)
                ++block_grid.z;

            if (block_grid.z > ((unsigned int)block_grid_max[2]))
            {
                TECA_ERROR("Too many blocks " << n_blocks << " of size " << block_size
                    << " are required for a grid of (" << block_grid_max[0] << ", "
                    << block_grid_max[1] << ", " << block_grid_max[2]
                    << ") blocks. Hint: increase the number of warps per block.");
                return -1;
            }
        }
        else
        {
            // 2d decomp
            block_grid.z = 1;
        }
    }
    else
    {
        // 1d decomp
        block_grid.x = n_blocks;
        block_grid.y = 1;
        block_grid.z = 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int partition_thread_blocks(int device_id, size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks,
    dim3 &thread_grid)
{
    int block_grid_max[3] = {0};
    int warp_size = 0;
    int warps_per_block_max = 0;
    if (get_launch_props(device_id, block_grid_max,
        warp_size, warps_per_block_max))
    {
        TECA_ERROR("Failed to get launch properties")
        return -1;
    }

    return partition_thread_blocks(array_size, warps_per_block,
        warp_size, block_grid_max, block_grid, n_blocks,
        thread_grid);
}

}
