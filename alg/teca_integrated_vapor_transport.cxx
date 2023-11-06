#include "teca_integrated_vapor_transport.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"
#include "teca_valid_value_mask.h"
#include "teca_array_attributes.h"
#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

#if defined(TECA_HAS_CUDA)
namespace cuda_gpu
{
// **************************************************************************
template <typename coord_t>
__global__
void plev_interfaces(coord_t *plev_int, const coord_t *plev,
    unsigned long nz, coord_t p_surf, coord_t p_top)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i == 0)
        plev_int[i] = p_surf;
    else if (i == nz)
        plev_int[i] = p_top;
    else if (i < nz)
        plev_int[i] = (plev[i-1] + plev[i]) / coord_t(2);
}

// **************************************************************************
__global__
void compute_mask(char *mask, const char *wind_valid,
    const char *q_valid, unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    mask[i] = (wind_valid[i] && q_valid[i] ? char(1) : char(0));
}

// **************************************************************************
template <typename num_t>
__global__
void compute_flux(num_t *f, const num_t *wind,
    const num_t *q, unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    f[i] = wind[i]*q[i];
}

// **************************************************************************
template <typename num_t>
__global__
void compute_flux(num_t *f, const num_t *wind,
    const num_t *q, const char *mask, unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    f[i] = (mask[i] ? wind[i]*q[i] : num_t(0));
}

// **************************************************************************
template <typename num_t, typename coord_t>
__global__
void compute_ivt(num_t *ivt, const num_t *flux,
    const coord_t *plev, unsigned long nxy, unsigned long nz,
    unsigned long stride)
{
    // NOTE: plev is specified at the level interfaces and the array is one element longer.

    // get the index into the data
    unsigned long i = 0;
    unsigned long k0 = 0;
    teca_cuda_util::thread_id_to_array_index_slab(i, k0, stride);

    // check bounds
    if ((i >= nxy) || (k0 >= nz))
        return;

    // get the upper loop bounds
    unsigned long k1 = k0 + stride;

    if (k1 > nz)
        k1 = nz;

    // integrate ivt over the vertical dimension
    num_t ivt_i = num_t();

    for (unsigned long q = k0; q < k1; ++q)
    {
        // dp over the slice
        num_t dp = plev[q + 1] - plev[q];

        // accumulate this partial column of data
        ivt_i += dp * flux[q*nxy + i];
    }

    atomicAdd(&ivt[i], ivt_i);
}

// **************************************************************************
template <typename num_t, typename coord_t>
__global__
void compute_ivt(num_t *ivt, const num_t *flux, const char *mask,
    const coord_t *plev, unsigned long nxy, unsigned long nz,
    unsigned long stride)
{
    // NOTE: plev is specified at the level interfaces and the array is one element longer.

    // get the index into the data
    unsigned long i = 0;
    unsigned long k0 = 0;
    teca_cuda_util::thread_id_to_array_index_slab(i, k0, stride);

    // check bounds
    if ((i >= nxy) || (k0 >= nz))
        return;

    // get the upper loop bounds
    unsigned long k1 = k0 + stride;

    if (k1 > nz)
        k1 = nz;

    // integrate ivt over the vertical dimension
    num_t ivt_i = num_t();

    for (unsigned long q = k0; q < k1; ++q)
    {
        // layer thickness
        num_t dp = plev[q + 1] - plev[q];

        // index into the column above i
        unsigned long qq = q*nxy + i;

        // accumulate
        ivt_i += (mask[qq] ? dp * flux[qq] : num_t(0));
    }

    atomicAdd(&ivt[i], ivt_i);
}

// **************************************************************************
template <typename num_t, typename coord_t>
__global__
void compute_ivt_trap(num_t *ivt, const num_t *flux,
    const coord_t *plev, unsigned long nxy, unsigned long nz,
    unsigned long stride)
{
    // get the index into the data
    unsigned long i = 0;
    unsigned long k0 = 0;
    teca_cuda_util::thread_id_to_array_index_slab(i, k0, stride);

    // check bounds
    if ((i >= nxy) || (k0 >= nz))
        return;

    // get the upper loop bounds
    unsigned long k1 = k0 + stride;

    if (k1 >= nz)
        k1 = nz - 1;

    // integrate ivt over the vertical dimension
    num_t ivt_i = num_t();

    for (unsigned long q = k0; q < k1; ++q)
    {
        unsigned long q1 = q + 1;

        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[q1] - plev[q]);

        // the current two x-y-planes of data
        unsigned long qq0 = q*nxy + i;
        unsigned long qq1 = q1*nxy + i;

        // accumulate this plane of data using trapezoid rule
        ivt_i += h2 * (flux[qq0] + flux[qq1]);
    }

    atomicAdd(&ivt[i], ivt_i);
}

// **************************************************************************
template <typename num_t, typename coord_t>
__global__
void compute_ivt_trap(num_t *ivt, const num_t *flux, const char *mask,
    const coord_t *plev, unsigned long nxy, unsigned long nz,
    unsigned long stride)
{
    // get the index into the data
    unsigned long i = 0;
    unsigned long k0 = 0;
    teca_cuda_util::thread_id_to_array_index_slab(i, k0, stride);

    // check bounds
    if ((i >= nxy) || (k0 >= nz))
        return;

    // get the upper loop bounds
    unsigned long k1 = k0 + stride;

    if (k1 >= nz)
        k1 = nz - 1;

    // integrate ivt over the vertical dimension
    num_t ivt_i = num_t();

    for (unsigned long q = k0; q < k1; ++q)
    {
        unsigned long q1 = q + 1;

        // dp over this vertical slice
        num_t h2 = num_t(0.5) * (plev[q1] - plev[q]);

        // the current two x-y-planes of data
        unsigned long qq0 = q*nxy + i;
        unsigned long qq1 = q1*nxy + i;

        // accumulate this plane of data using trapezoid rule
        ivt_i += (mask[qq0] && mask[qq1] ?
             h2 * (flux[qq0] + flux[qq1]) : num_t(0));
    }

    atomicAdd(&ivt[i], ivt_i);
}

// **************************************************************************
template <typename num_t, typename coord_t>
__global__
void scale_ivt(num_t *ivt, const coord_t *plev, unsigned long nxy)
{
    // index into the 2D output data
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= nxy)
        return;

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);

    ivt[i] *= m1g;
}

// **************************************************************************
template <typename coord_t, typename num_t>
int cartesian_ivt_trap(int device_id, unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind, const num_t *q,
    num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // determine flux kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nxyz, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute the flux
    hamr::buffer<num_t> flux(hamr::buffer_allocator::cuda_async, nxyz);
    num_t *pflux = flux.data();

    cudaError_t ierr = cudaSuccess;
    compute_flux<<<block_grid,thread_grid>>>(pflux, wind, q, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_flux CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine ivt kernel launch parameters.
    block_grid = 0;
    thread_grid = 0;
    int n_blocks_xy = 0;
    int n_blocks_z = 0;
    size_t stride = 32;
    if (teca_cuda_util::partition_thread_blocks_slab(device_id,
        nxy, nz - 1, stride, 8, block_grid, n_blocks_xy, n_blocks_z,
        thread_grid))
    {
        TECA_ERROR("Failed to slab partition thread blocks")
        return -1;
    }

    // calculate ivt
    compute_ivt_trap<<<block_grid,thread_grid>>>(ivt, pflux, plev, nxy, nz, stride);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine scale kernel launch parameters
    block_grid.y = 1;

    // scale the result
    scale_ivt<<<block_grid,thread_grid>>>(ivt, plev, nxy);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the scale_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}


// **************************************************************************
template <typename coord_t, typename num_t>
int cartesian_ivt(int device_id, unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const num_t *q, num_t *ivt, coord_t p_surf, coord_t p_top)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;
    unsigned long nz1 = nz + 1;

    // determine plev interface kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nz1, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute pressure level interfaces
    hamr::buffer<coord_t> plev_int(hamr::buffer_allocator::cuda_async, nz1);
    coord_t *pplev_int = plev_int.data();

    cudaError_t ierr = cudaSuccess;
    plev_interfaces<<<block_grid,thread_grid>>>(pplev_int, plev, nz, p_surf, p_top);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the plev_interfaces CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine flux kernel launch parameters
    n_blocks = 0;
    block_grid = 0;
    thread_grid = 0;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nxyz, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute the flux
    hamr::buffer<num_t> flux(hamr::buffer_allocator::cuda_async, nxyz);
    num_t *pflux = flux.data();

    ierr = cudaSuccess;
    compute_flux<<<block_grid,thread_grid>>>(pflux, wind, q, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_flux CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine ivt kernel launch parameters.
    block_grid = 0;
    thread_grid = 0;
    int n_blocks_xy = 0;
    int n_blocks_z = 0;
    size_t stride = 32;
    if (teca_cuda_util::partition_thread_blocks_slab(device_id,
        nxy, nz, stride, 8, block_grid, n_blocks_xy, n_blocks_z,
        thread_grid))
    {
        TECA_ERROR("Failed to slab partition thread blocks")
        return -1;
    }

    // calculate ivt
    compute_ivt<<<block_grid,thread_grid>>>(ivt, pflux, pplev_int, nxy, nz, stride);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine scale kernel launch parameters
    block_grid.y = 1;

    // scale the result
    scale_ivt<<<block_grid,thread_grid>>>(ivt, plev, nxy);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the scale_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// **************************************************************************
template <typename coord_t, typename num_t>
int cartesian_ivt(int device_id, unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const char *wind_valid, const num_t *q, const char *q_valid,
    num_t *ivt, coord_t p_surf, coord_t p_top)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;
    unsigned long nz1 = nz + 1;

    // determine plev interface kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nz1, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute pressure level interfaces
    hamr::buffer<coord_t> plev_int(hamr::buffer_allocator::cuda_async, nz1);
    coord_t *pplev_int = plev_int.data();

    cudaError_t ierr = cudaSuccess;
    plev_interfaces<<<block_grid,thread_grid>>>(pplev_int, plev, nz, p_surf, p_top);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the plev_interfaces CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine flux and mask kernel launch parameters
    block_grid = 0;
    thread_grid = 0;
    n_blocks = 0;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nxyz, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute the mask
    hamr::buffer<char> mask(hamr::buffer_allocator::cuda_async, nxyz);
    char *pmask = mask.data();

    compute_mask<<<block_grid,thread_grid>>>(pmask, wind_valid, q_valid, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_mask CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // compute the flux
    hamr::buffer<num_t> flux(hamr::buffer_allocator::cuda_async, nxyz);
    num_t *pflux = flux.data();

    compute_flux<<<block_grid,thread_grid>>>(pflux, wind, q, pmask, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the flux CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine ivt kernel launch parameters.
    block_grid = 0;
    thread_grid = 0;
    size_t stride = 32;
    int n_blocks_xy = 0;
    int n_blocks_z = 0;
    if (teca_cuda_util::partition_thread_blocks_slab(device_id,
        nxy, nz, stride, 8, block_grid, n_blocks_xy, n_blocks_z,
        thread_grid))
    {
        TECA_ERROR("Failed to slab partition thread blocks")
        return -1;
    }

    // calculate ivt
    compute_ivt<<<block_grid,thread_grid>>>(ivt,
        pflux, pmask, pplev_int, nxy, nz, stride);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine scale kernel launch parameters
    block_grid.y = 1;

    // scale the result
    scale_ivt<<<block_grid,thread_grid>>>(ivt, plev, nxy);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the scale_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// **************************************************************************
template <typename coord_t, typename num_t>
int cartesian_ivt_trap(int device_id, unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const char *wind_valid, const num_t *q, const char *q_valid,
    num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // determine flux and mask kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        nxyz, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // compute the mask
    hamr::buffer<char> mask(hamr::buffer_allocator::cuda_async, nxyz);
    char *pmask = mask.data();

    cudaError_t ierr = cudaSuccess;
    compute_mask<<<block_grid,thread_grid>>>(pmask, wind_valid, q_valid, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_mask CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // compute the flux
    hamr::buffer<num_t> flux(hamr::buffer_allocator::cuda_async, nxyz);
    num_t *pflux = flux.data();

    compute_flux<<<block_grid,thread_grid>>>(pflux, wind, q, pmask, nxyz);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the flux CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine ivt kernel launch parameters.
    block_grid = 0;
    thread_grid = 0;
    size_t stride = 32;
    int n_blocks_xy = 0;
    int n_blocks_z = 0;
    if (teca_cuda_util::partition_thread_blocks_slab(device_id,
        nxy, nz - 1, stride, 8, block_grid, n_blocks_xy, n_blocks_z,
        thread_grid))
    {
        TECA_ERROR("Failed to slab partition thread blocks")
        return -1;
    }

    // calculate ivt
    compute_ivt_trap<<<block_grid,thread_grid>>>(ivt,
        pflux, pmask, plev, nxy, nz, stride);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    // determine scale kernel launch parameters
    block_grid.y = 1;

    // scale the result
    scale_ivt<<<block_grid,thread_grid>>>(ivt, plev, nxy);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the scale_ivt CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// **************************************************************************
int dispatch(int device_id, size_t nx, size_t ny, size_t nz,
    const const_p_teca_variant_array &p,
    const const_p_teca_variant_array &wind_u,
    const const_p_teca_variant_array &wind_u_valid,
    const const_p_teca_variant_array &wind_v,
    const const_p_teca_variant_array &wind_v_valid,
    const const_p_teca_variant_array &q,
    const const_p_teca_variant_array &q_valid,
    p_teca_variant_array &ivt_u,
    p_teca_variant_array &ivt_v,
    double p_surf, double p_top,
    int use_trapezoid_rule)
{
    // set the CUDA device to run on
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
    {
        TECA_ERROR("Failed to set the CUDA device to " << device_id
            << ". " << cudaGetErrorString(ierr))
        return -1;
    }

    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        p.get(), _COORDS,

        auto [sp_p, p_p] = get_cuda_accessible<CTT_COORDS>(p);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            wind_u.get(), _DATA,

            // allocate and initialize the ivt output to zero
            unsigned long nxy = nx*ny;

            NT_DATA *p_ivt_u = nullptr;
            std::tie(ivt_u, p_ivt_u) = ::New<TT_DATA>(nxy, NT_DATA(), allocator::cuda_async);

            NT_DATA *p_ivt_v = nullptr;
            std::tie(ivt_v, p_ivt_v) = ::New<TT_DATA>(nxy, NT_DATA(), allocator::cuda_async);

            // get the inputs
            assert_type<CTT_DATA>(wind_v, q);
            auto [sp_wu, p_wind_u] = get_cuda_accessible<CTT_DATA>(wind_u);
            auto [sp_wv, p_wind_v] = get_cuda_accessible<CTT_DATA>(wind_v);
            auto [sp_q, p_q] = get_cuda_accessible<CTT_DATA>(q);

            if (wind_u_valid)
            {
                auto [sp_wuv, p_wind_u_valid] = get_cuda_accessible<CTT_MASK>(wind_u_valid);
                auto [sp_wvv, p_wind_v_valid] = get_cuda_accessible<CTT_MASK>(wind_v_valid);
                auto [sp_qv, p_q_valid] = get_cuda_accessible<CTT_MASK>(q_valid);

                int ierr_u = 0, ierr_v = 0;
                if (use_trapezoid_rule)
                {
                    ierr_u = cuda_gpu::cartesian_ivt_trap(device_id, nx, ny, nz,
                        p_p, p_wind_u, p_wind_u_valid, p_q, p_q_valid, p_ivt_u);

                    ierr_v = cuda_gpu::cartesian_ivt_trap(device_id, nx, ny, nz,
                        p_p, p_wind_v, p_wind_v_valid, p_q, p_q_valid, p_ivt_v);
                }
                else
                {
                    ierr_u = cuda_gpu::cartesian_ivt(device_id, nx, ny, nz,
                        p_p, p_wind_u, p_wind_u_valid, p_q, p_q_valid, p_ivt_u,
                        NT_COORDS(p_surf), NT_COORDS(p_top));

                    ierr_v = cuda_gpu::cartesian_ivt(device_id, nx, ny, nz,
                        p_p, p_wind_v, p_wind_v_valid, p_q, p_q_valid, p_ivt_v,
                        NT_COORDS(p_surf), NT_COORDS(p_top));
                }

                if (ierr_u || ierr_v)
                {
                    TECA_ERROR("Failed to compute IVT with valid value mask")
                    return -1;
                }
            }
            else
            {
                int ierr_u = 0, ierr_v = 0;
                if (use_trapezoid_rule)
                {
                    ierr_u = cuda_gpu::cartesian_ivt_trap(device_id,
                        nx, ny, nz, p_p, p_wind_u, p_q, p_ivt_u);

                    ierr_v = cuda_gpu::cartesian_ivt_trap(device_id,
                        nx, ny, nz, p_p, p_wind_v, p_q, p_ivt_v);
                }
                else
                {
                    ierr_u = cuda_gpu::cartesian_ivt(device_id,
                        nx, ny, nz, p_p, p_wind_u, p_q, p_ivt_u,
                        NT_COORDS(p_surf), NT_COORDS(p_top));

                    ierr_v = cuda_gpu::cartesian_ivt(device_id,
                        nx, ny, nz, p_p, p_wind_v, p_q, p_ivt_v,
                        NT_COORDS(p_surf), NT_COORDS(p_top));
                }

                if (ierr_u || ierr_v)
                {
                    TECA_ERROR("Failed to compute IVT")
                    return -1;
                }
            }
            )
        )

    return 0;
}
}
#endif

namespace cpu
{
// **************************************************************************
template <typename coord_t>
void plev_interfaces(coord_t *p_int, const coord_t *p_cell,
    unsigned long nz, coord_t p0, coord_t p1)
{
    p_int[0] = p0;

    ++p_int;

    unsigned long nzm1 = nz - 1;
    for (unsigned long i = 0; i < nzm1; ++i)
    {
        p_int[i] = (p_cell[i] + p_cell[i + 1]) / coord_t(2);
    }

    p_int[nzm1] = p1;

#if defined(TECA_DEBUG)
    std::cerr << "p_int = [";
    --p_int;
    for (unsigned long i = 0; i <= nz; ++i)
        std::cerr << p_int[i] << ", ";
    std::cerr << "]" << std::endl;

    std::cerr << "dp = [";
    for (unsigned long i = 0; i < nz; ++i)
        std::cerr << p_int[i+1] - p_int[i]  << ", ";
    std::cerr << "]" << std::endl;
#endif
}

// **************************************************************************
template <typename coord_t, typename num_t>
void cartesian_ivt(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const num_t *q, num_t *ivt, coord_t p0, coord_t p1)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;
    unsigned long nz1 = nz + 1;

    // compute pressure level interfaces
    coord_t *plev_int = (coord_t*)malloc(nz1*sizeof(coord_t));
    plev_interfaces(plev_int, plev, nz, p0, p1);

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        f[i] = wind[i]*q[i];
    }

    // work an x-y slice at  a time
    for (unsigned long k = 0; k < nz; ++k)
    {
        // pressure level thickness
        num_t dp = plev_int[k+1] - plev_int[k];

        // the current x-y-plane of data
        unsigned long knxy = k*nxy;
        num_t *f_k = f + knxy;

        // accumulate this plane of data
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += dp * f_k[q];
        }
    }

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);

    for (unsigned long i = 0; i < nxy; ++i)
    {
        ivt[i] *= m1g;
    }

    // free up the integrand and coordinates
    free(f);
    free(plev_int);
}

// **************************************************************************
template <typename coord_t, typename num_t>
void cartesian_ivt(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const char *wind_valid, const num_t *q, const char *q_valid,
    num_t *ivt, coord_t p0, coord_t p1)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;
    unsigned long nz1 = nz + 1;

    // compute pressure level interfaces
    coord_t *plev_int = (coord_t*)malloc(nz1*sizeof(coord_t));
    plev_interfaces(plev_int, plev, nz, p0, p1);

    // compute the mask
    char *mask = (char*)malloc(nxyz);
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        mask[i] = (wind_valid[i] && q_valid[i] ? char(1) : char(0));
    }

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        f[i] = (mask[i] ? wind[i]*q[i] : num_t(0));
    }

    // work an x-y slice at a time
    for (unsigned long k = 0; k < nz; ++k)
    {
        // dp over the slice
        num_t dp = plev_int[k+1] - plev_int[k];

        // the current x-y-plane of data
        unsigned long knxy = k*nxy;
        char *mask_k = mask + knxy;
        num_t *f_k = f + knxy;

        // accumulate this plane of data
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += (mask_k[q] ?  dp * f_k[q] : num_t(0));
        }
    }

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);

    for (unsigned long i = 0; i < nxy; ++i)
    {
        ivt[i] *= m1g;
    }

    // free up the integrand, coordinates, and mask
    free(mask);
    free(plev_int);
    free(f);
}

// **************************************************************************
template <typename coord_t, typename num_t>
void cartesian_ivt_trap(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const num_t *q, num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        f[i] = wind[i]*q[i];
    }

    // work an x-y slice at  a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        num_t *f_k0 = f + knxy;
        num_t *f_k1 = f_k0 + nxy;

        // accumulate this plane of data using trapezoid rule
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += h2 * (f_k0[q] + f_k1[q]);
        }
    }

    // free up the integrand
    free(f);

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);

    for (unsigned long i = 0; i < nxy; ++i)
    {
        ivt[i] *= m1g;
    }
}

// **************************************************************************
template <typename coord_t, typename num_t>
void cartesian_ivt_trap(unsigned long nx, unsigned long ny,
    unsigned long nz, const coord_t *plev, const num_t *wind,
    const char *wind_valid, const num_t *q, const char *q_valid,
    num_t *ivt)
{
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // compute the mask
    char *mask = (char*)malloc(nxyz);
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        mask[i] = (wind_valid[i] && q_valid[i] ? char(1) : char(0));
    }

    // compute the integrand
    num_t *f = (num_t*)malloc(nxyz*sizeof(num_t));
    for (unsigned long i = 0; i < nxyz; ++i)
    {
        f[i] = (mask[i] ? wind[i]*q[i] : num_t(0));
    }

    // work an x-y slice at a time
    unsigned long nzm1 = nz - 1;
    for (unsigned long k = 0; k < nzm1; ++k)
    {
        // dp over the slice
        num_t h2 = num_t(0.5) * (plev[k+1] - plev[k]);

        // the current two x-y-planes of data
        unsigned long knxy = k*nxy;
        num_t *f_k0 = f + knxy;
        num_t *f_k1 = f_k0 + nxy;

        char *mask_k0 = mask + knxy;
        char *mask_k1 = mask_k0 + nxy;

        // accumulate this plane of data using trapezoid rule
        for (unsigned long q = 0; q < nxy; ++q)
        {
            ivt[q] += ((mask_k0[q] && mask_k1[q]) ?
               h2 * (f_k0[q] + f_k1[q]) : num_t(0));
        }
    }

    // free up the integrand and mask
    free(mask);
    free(f);

    // check the sign, in this way we can handle both increasing and decreasing
    // pressure coordinates
    num_t s = plev[1] - plev[0] < num_t(0) ? num_t(-1) : num_t(1);

    // scale by -1/g
    num_t m1g = s/num_t(9.80665);

    for (unsigned long i = 0; i < nxy; ++i)
    {
        ivt[i] *= m1g;
    }
}

// **************************************************************************
int dispatch(size_t nx, size_t ny, size_t nz,
    const const_p_teca_variant_array &p,
    const const_p_teca_variant_array &wind_u,
    const const_p_teca_variant_array &wind_u_valid,
    const const_p_teca_variant_array &wind_v,
    const const_p_teca_variant_array &wind_v_valid,
    const const_p_teca_variant_array &q,
    const const_p_teca_variant_array &q_valid,
    p_teca_variant_array &ivt_u,
    p_teca_variant_array &ivt_v,
    double p0, double p1,
    int use_trapezoid_rule)
{
    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        p.get(), _COORDS,

        auto [sp_p, p_p] = get_host_accessible<CTT_COORDS>(p);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            wind_u.get(), _DATA,

            // allocate and initialize the ivt output to zero
            unsigned long nxy = nx*ny;

            NT_DATA *p_ivt_u = nullptr;
            std::tie(ivt_u, p_ivt_u) = ::New<TT_DATA>(nxy, NT_DATA());

            NT_DATA *p_ivt_v = nullptr;
            std::tie(ivt_v, p_ivt_v) = ::New<TT_DATA>(nxy, NT_DATA());

            auto [sp_wu, p_wind_u] = get_host_accessible<CTT_DATA>(wind_u);
            auto [sp_wv, p_wind_v] = get_host_accessible<CTT_DATA>(wind_v);
            auto [sp_q, p_q] = get_host_accessible<CTT_DATA>(q);

            if (wind_u_valid)
            {
                assert_type<CTT_MASK>(wind_u_valid, wind_v_valid, q_valid);
                auto [sp_wuv, p_wind_u_valid] = get_host_accessible<CTT_MASK>(wind_u_valid);
                auto [sp_wvv, p_wind_v_valid] = get_host_accessible<CTT_MASK>(wind_v_valid);
                auto [sp_qv, p_q_valid] = get_host_accessible<CTT_MASK>(q_valid);

                sync_host_access_any(p, wind_u, wind_v, q,
                                     wind_u_valid, wind_v_valid, q_valid);

                if (use_trapezoid_rule)
                {
                    cpu::cartesian_ivt_trap(nx, ny, nz, p_p,
                        p_wind_u, p_wind_u_valid, p_q, p_q_valid, p_ivt_u);

                    cpu::cartesian_ivt_trap(nx, ny, nz, p_p,
                        p_wind_v, p_wind_v_valid, p_q, p_q_valid, p_ivt_v);
                }
                else
                {
                    cpu::cartesian_ivt(nx, ny, nz, p_p, p_wind_u,
                        p_wind_u_valid, p_q, p_q_valid, p_ivt_u,
                        NT_COORDS(p0), NT_COORDS(p1));

                    cpu::cartesian_ivt(nx, ny, nz, p_p, p_wind_v,
                        p_wind_v_valid, p_q, p_q_valid, p_ivt_v,
                        NT_COORDS(p0), NT_COORDS(p1));
                }
            }
            else
            {
                sync_host_access_any(p, wind_u, wind_v, q);

                if (use_trapezoid_rule)
                {
                    cpu::cartesian_ivt_trap(nx, ny, nz, p_p, p_wind_u, p_q, p_ivt_u);
                    cpu::cartesian_ivt_trap(nx, ny, nz, p_p, p_wind_v, p_q, p_ivt_v);
                }
                else
                {
                    cpu::cartesian_ivt(nx, ny, nz, p_p, p_wind_u, p_q, p_ivt_u,
                        NT_COORDS(p0), NT_COORDS(p1));

                    cpu::cartesian_ivt(nx, ny, nz, p_p, p_wind_v, p_q, p_ivt_v,
                        NT_COORDS(p0), NT_COORDS(p1));
                }
            }
            )
        )

    return 0;
}
}

// --------------------------------------------------------------------------
teca_integrated_vapor_transport::teca_integrated_vapor_transport() :
    wind_u_variable("ua"), wind_v_variable("va"),
    specific_humidity_variable("hus"), ivt_u_variable("ivt_u"),
    ivt_v_variable("ivt_v"), fill_value(1.0e20), surface_pressure(101325),
    top_pressure(0), use_trapezoid_rule(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_integrated_vapor_transport::~teca_integrated_vapor_transport()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_integrated_vapor_transport::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_integrated_vapor_transport":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, wind_u_variable,
            "name of the variable containg the lon component of the wind vector")
        TECA_POPTS_GET(std::string, prefix, wind_v_variable,
            "name of the variable containg the lat component of the wind vector")
        TECA_POPTS_GET(std::string, prefix, specific_humidity_variable,
            "name of the variable containg the specific humidity")
        TECA_POPTS_GET(int, prefix, use_trapezoid_rule,
            "selects between trapezoid rule and a first order scheme.")
        TECA_POPTS_GET(double, prefix, surface_pressure,
            "set the surface pressure for first order integration")
        TECA_POPTS_GET(double, prefix, top_pressure,
            "set the top pressure for first order integration")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_integrated_vapor_transport::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, wind_u_variable)
    TECA_POPTS_SET(opts, std::string, prefix, wind_v_variable)
    TECA_POPTS_SET(opts, std::string, prefix, specific_humidity_variable)
    TECA_POPTS_SET(opts, int, prefix, use_trapezoid_rule)
    TECA_POPTS_SET(opts, double, prefix, surface_pressure)
    TECA_POPTS_SET(opts, double, prefix, top_pressure)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_integrated_vapor_transport::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_vapor_transport::get_output_metadata" << std::endl;
#endif
    (void)port;

    // set things up in the first pass, and don't modify in subsequent passes
    // due to threading concerns

    if (this->get_number_of_derived_variables() == 0)
    {
        // the base class will handle dealing with the transformation of
        // mesh dimensions and reporting the array we produce, but we have
        // to determine the data type and tell the name of the produced array.
        const teca_metadata &md = input_md[0];

        teca_metadata attributes;
        if (md.get("attributes", attributes))
        {
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes are misisng")
            return teca_metadata();
        }

        teca_metadata u_atts;
        if (attributes.get(this->wind_u_variable, u_atts))
        {
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->wind_u_variable
                << "\" are misisng")
            return teca_metadata();
        }

        int type_code = 0;
        if (u_atts.get("type_code", type_code))
        {
            TECA_FATAL_ERROR("Failed to determine output data type "
                "because attributes for \"" << this->wind_u_variable
                << "\" is misisng a \"type_code\"")
            return teca_metadata();
        }

        teca_array_attributes ivt_u_atts(
            type_code, teca_array_attributes::point_centering,
            0, teca_array_attributes::xyt_active(), "kg m-1 s-1",
            "longitudinal integrated vapor transport",
            "the longitudinal component of integrated vapor transport",
            1, this->fill_value);

        teca_array_attributes ivt_v_atts(
            type_code, teca_array_attributes::point_centering,
            0, teca_array_attributes::xyt_active(), "kg m-1 s-1",
            "latitudinal integrated vapor transport",
            "the latitudinal component of integrated vapor transport",
            this->fill_value);

        // install name and attributes of the output variables in the base classs
        this->append_derived_variable(this->ivt_u_variable);
        this->append_derived_variable(this->ivt_v_variable);

        this->append_derived_variable_attribute(ivt_u_atts);
        this->append_derived_variable_attribute(ivt_v_atts);

    }

    if (this->get_number_of_dependent_variables() == 0)
    {
        // install the names of the input variables in the base class
        this->append_dependent_variable(this->wind_u_variable);
        this->append_dependent_variable(this->wind_v_variable);
        this->append_dependent_variable(this->specific_humidity_variable);
    }

    // invoke the base class method, which does the work of transforming
    // the mesh and reporting the variables and their attributes.
    return teca_vertical_reduction::get_output_metadata(port, input_md);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_integrated_vapor_transport::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    // invoke the base class method
    return teca_vertical_reduction::get_upstream_request(port, input_md, request);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_integrated_vapor_transport::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_integrated_vapor_transport::execute" << std::endl;
#endif
    (void)port;

    int rank = 0;

#if defined(TECA_HAS_MPI)
    MPI_Comm comm = this->get_communicator();

    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(comm, &rank);
#endif

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because a cartesian mesh is required.")
        return nullptr;
    }

    // get the input dimensions
    unsigned long extent[6] = {0};
    if (in_mesh->get_extent(extent))
    {
        TECA_FATAL_ERROR("Failed to compute IVT because mesh extent is missing.")
        return nullptr;
    }

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    // get the pressure coordinates
    const_p_teca_variant_array p = in_mesh->get_z_coordinates();
    if (!p)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because pressure coordinates are missing")
        return nullptr;
    }

    if (p->size() < 2)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because z dimensions "
            << p->size() << " < 2 as required by the integration method")
        return nullptr;
    }

    // assume descending vertical coordinates. the integral is coded to handle
    // either however when using the first order method the surface and top
    // level pressures must be switched when the coordinates are ascending.
    double p0 = this->surface_pressure;
    double p1 = this->top_pressure;
    if (!this->use_trapezoid_rule)
    {
        double z0 = 0.0;
        double z1 = 0.0;

        p->get(0, z0);
        p->get(nz - 1, z1);

        // check for ascendinig or descending vertical axis.
        // flip the surface and top level pressures to handle ascending
        // vertical coordinates
        if (z1 > z0)
        {
            p0 = this->top_pressure;
            p1 = this->surface_pressure;
        }
    }

    // get the attributes
    teca_metadata in_atts;
    in_mesh->get_attributes(in_atts);

    // verify that units of the vertical axis are Pa
    if (rank == 0)
    {
        teca_metadata z_axis_atts;
        std::string z_axis_variable, z_axis_units;
        if (in_mesh->get_z_coordinate_variable(z_axis_variable) ||
            in_atts.get(z_axis_variable, z_axis_atts)   ||
            z_axis_atts.get("units", z_axis_units))
        {
            TECA_WARNING("Metadata issue, failed to get attributes"
                " for the z_axis_variable \"" << z_axis_variable
                << "\". Units check will be skipped.")
        }
        else if (z_axis_units != "Pa")
        {
            TECA_WARNING("Invalid vertical coordinate units. \""
                << z_axis_variable << "\" has units \"" << z_axis_units
                << "\" but units of Pa are required. IVT will be"
                " calculated but the result may be incorrect.")
        }
    }

    // gather the input arrays
    const_p_teca_variant_array wind_u =
        in_mesh->get_point_arrays()->get(this->wind_u_variable);

    if (!wind_u)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because longitudinal wind \""
            << this->wind_u_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array wind_u_valid =
           in_mesh->get_point_arrays()->get(this->wind_u_variable + "_valid");

    const_p_teca_variant_array wind_v =
        in_mesh->get_point_arrays()->get(this->wind_v_variable);

    if (!wind_v)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because latitudinal wind \""
            << this->wind_v_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array wind_v_valid =
           in_mesh->get_point_arrays()->get(this->wind_v_variable + "_valid");

    const_p_teca_variant_array q =
        in_mesh->get_point_arrays()->get(this->specific_humidity_variable);

    if (!q)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because specific humidity \""
            << this->specific_humidity_variable << "\" is missing")
        return nullptr;
    }

    const_p_teca_variant_array q_valid =
           in_mesh->get_point_arrays()->get(this->specific_humidity_variable + "_valid");

    // the base class will construct the output mesh
    p_teca_cartesian_mesh out_mesh
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(
                teca_vertical_reduction::execute(port, input_data, request)));

    if (!out_mesh)
    {
        TECA_FATAL_ERROR("Failed to compute IVT because the output mesh was "
            "not constructed")
        return nullptr;
    }

    // calculate IVT
    p_teca_variant_array ivt_u;
    p_teca_variant_array ivt_v;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (cuda_gpu::dispatch(device_id, nx, ny, nz, p, wind_u,
            wind_u_valid, wind_v, wind_v_valid, q, q_valid, ivt_u,
            ivt_v, p0, p1, this->use_trapezoid_rule))
        {
            TECA_ERROR("Failed to compute IVT using CUDA")
            return nullptr;
        }
    }
    else
    {
#endif
        if (cpu::dispatch(nx, ny, nz, p, wind_u, wind_u_valid,
            wind_v, wind_v_valid, q, q_valid, ivt_u, ivt_v,
            p0, p1, this->use_trapezoid_rule))
        {
            TECA_ERROR("Failed to compute IVT on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // store the result
    out_mesh->get_point_arrays()->set(this->ivt_u_variable, ivt_u);
    out_mesh->get_point_arrays()->set(this->ivt_v_variable, ivt_v);

    // pass 2D arrays through.
    p_teca_array_collection in_arrays =
        std::const_pointer_cast<teca_array_collection>(in_mesh->get_point_arrays());

    p_teca_array_collection out_arrays = out_mesh->get_point_arrays();

    unsigned long nxy = nx*ny;
    int n_arrays = in_arrays->size();
    for (int i = 0; i < n_arrays; ++i)
    {
        p_teca_variant_array array = in_arrays->get(i);
        if (array->size() == nxy)
        {
            // pass the array.
            out_arrays->append(in_arrays->get_name(i), array);
        }
    }

    return out_mesh;
}
