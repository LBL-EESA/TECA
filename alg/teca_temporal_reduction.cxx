#include "teca_temporal_reduction.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_metadata.h"
#include "teca_calendar_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

#define TECA_DEBUG

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

// --------------------------------------------------------------------------
void reduction_operator_collection::initialize(double fill_value)
{
    this->fill_value = fill_value;
}

// --------------------------------------------------------------------------
void average_operator::initialize(double fill_value)
{
    this->fill_value = fill_value;
    this->count = nullptr;
}

#if defined(TECA_HAS_CUDA)
namespace cuda
{
// --------------------------------------------------------------------------
template <typename T>
__global__
void count_init(const char *p_out_valid, T *p_count, unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_out_valid == nullptr)
    {
        p_count[i] = 1.;
    }
    else
    {
        p_count[i] = (p_out_valid[i]) ? 1. : 0.;
    }
}

// --------------------------------------------------------------------------
template <typename T>
int count_init(int device_id, const char *p_out_valid, T *p_count,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    count_init<<<block_grid,thread_grid>>>(p_out_valid, p_count, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void average(
     const T *p_out_array, const char *p_out_valid,
     const T *p_in_array, const char *p_in_valid,
     T *p_count,
     T *p_red_array, char *p_red_valid,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_red_valid[i] = char(1);
    if (p_in_valid == nullptr)
    {
        // update count
        p_count[i] += 1.;

        // accumulate
        p_red_array[i] = p_out_array[i] + p_in_array[i];
    }
    else
    {
        //update the count only where there is valid data
        if (p_in_valid[i])
        {
            p_count[i] += 1.;
        }

        if (p_in_valid[i] && p_out_valid[i])
        {
            // accumulate
            p_red_array[i] = p_out_array[i] + p_in_array[i];
        }
        else if (p_in_valid[i] && !p_out_valid[i])
        {
            p_red_array[i] = p_in_array[i];
        }
        else if (!p_in_valid[i] && p_out_valid[i])
        {
            p_red_array[i] = p_out_array[i];
        }
        else
        {
            // update the valid value mask
            p_red_valid[i] = char(0);
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int average(
    int device_id,
    const T *p_out_array, const char *p_out_valid,
    const T *p_in_array, const char *p_in_valid,
    T *p_count,
    T *p_red_array, char *p_red_valid,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    average<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
         p_in_array, p_in_valid, p_count, p_red_array, p_red_valid, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void summation(
     const T *p_out_array, const char *p_out_valid,
     const T *p_in_array, const char *p_in_valid,
     T *p_red_array, char *p_red_valid,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_red_valid[i] = char(1);
    if (p_in_valid == nullptr)
    {
        // accumulate
        p_red_array[i] = p_out_array[i] + p_in_array[i];
    }
    else
    {
        if (p_in_valid[i] && p_out_valid[i])
        {
            // accumulate
            p_red_array[i] = p_out_array[i] + p_in_array[i];
        }
        else if (p_in_valid[i] && !p_out_valid[i])
        {
            p_red_array[i] = p_in_array[i];
        }
        else if (!p_in_valid[i] && p_out_valid[i])
        {
            p_red_array[i] = p_out_array[i];
        }
        else
        {
            // update the valid value mask
            p_red_valid[i] = char(0);
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int summation(
    int device_id,
    const T *p_out_array, const char *p_out_valid,
    const T *p_in_array, const char *p_in_valid,
    T *p_red_array, char *p_red_valid,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    summation<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
           p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}


// --------------------------------------------------------------------------
template <typename T>
__global__
void minimum(
     const T *p_out_array, const char *p_out_valid,
     const T *p_in_array, const char *p_in_valid,
     T *p_red_array, char *p_red_valid,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_red_valid[i] = char(1);
    if (p_in_valid == nullptr)
    {
        // reduce
        p_red_array[i] = (p_in_array[i] < p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
    }
    else
    {
        if (p_in_valid[i] && p_out_valid[i])
        {
            // reduce
            p_red_array[i] = (p_in_array[i] < p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
        }
        else if (p_in_valid[i] && !p_out_valid[i])
        {
            p_red_array[i] = p_in_array[i];
        }
        else if (!p_in_valid[i] && p_out_valid[i])
        {
            p_red_array[i] = p_out_array[i];
        }
        else
        {
            // update the valid value mask
            p_red_valid[i] = char(0);
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int minimum(
    int device_id,
    const T *p_out_array, const char *p_out_valid,
    const T *p_in_array, const char *p_in_valid,
    T *p_red_array, char *p_red_valid,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    minimum<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
         p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void maximum(
     const T *p_out_array, const char *p_out_valid,
     const T *p_in_array, const char *p_in_valid,
     T *p_red_array, char *p_red_valid,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    p_red_valid[i] = char(1);
    if (p_in_valid == nullptr)
    {
        // reduce
        p_red_array[i] = (p_in_array[i] > p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
    }
    else
    {
        if (p_in_valid[i] && p_out_valid[i])
        {
            // reduce
            p_red_array[i] = (p_in_array[i] > p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
        }
        else if (p_in_valid[i] && !p_out_valid[i])
        {
            p_red_array[i] = p_in_array[i];
        }
        else if (!p_in_valid[i] && p_out_valid[i])
        {
            p_red_array[i] = p_out_array[i];
        }
        else
        {
            // update the valid value mask
            p_red_valid[i] = char(0);
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int maximum(
    int device_id,
    const T *p_out_array, const char *p_out_valid,
    const T *p_in_array, const char *p_in_valid,
    T *p_red_array, char *p_red_valid,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    maximum<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
         p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void finalize(
     T *p_out_array, const char *p_out_valid,
     T *p_red_array,
     const T *p_count,
     double fill_value,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_out_valid == nullptr)
    {
        p_red_array[i] = p_out_array[i]/p_count[i];
    }
    else
    {
        p_red_array[i] = (!p_out_valid[i]) ?
                                 fill_value : p_out_array[i]/p_count[i];
    }
}

// --------------------------------------------------------------------------
template <typename T>
int finalize(
    int device_id,
    T *p_out_array, const char *p_out_valid,
    T *p_red_array,
    const T *p_count,
    double fill_value,
    unsigned long n_elem)
{
    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(device_id,
        n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        TECA_ERROR("Failed to partition thread blocks")
        return -1;
    }

    // launch the finalize kernel
    cudaError_t ierr = cudaSuccess;
    finalize<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
          p_red_array, p_count, fill_value, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}
}
#endif

// --------------------------------------------------------------------------
int average_operator::update(
    int device_id,
    const const_p_teca_variant_array &out_array,
    const const_p_teca_variant_array &out_valid,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    p_teca_variant_array &red_array,
    p_teca_variant_array &red_valid)
{
    unsigned long n_elem = out_array->size();

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        // set the CUDA device to run on
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
        {
            TECA_ERROR("Failed to set the CUDA device to " << device_id
                << ". " << cudaGetErrorString(ierr))
            return -1;
        }

        red_array = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);
        red_valid = out_valid->new_instance(n_elem, teca_variant_array::allocator::cuda);
    }
    else
    {
#endif
        red_array = out_array->new_instance(n_elem);
        red_valid = out_valid->new_instance(n_elem);
#if defined(TECA_HAS_CUDA)
    }
#endif

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        red_array.get(),

        using NT_MASK = char;
        using TT_MASK = teca_variant_array_impl<NT_MASK>;

        // initialize the count the first time through; this needs to
        // happen now since before this we don't know where invalid
        // values are.
        if (this->count == nullptr)
        {
#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                this->count = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);

                auto sp_count = static_cast<TT*>(this->count.get())->get_cuda_accessible();
                NT *p_count = sp_count.get();

                if (this->fill_value != -1)
                {
                    auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                    const NT_MASK *p_out_valid = sp_out_valid.get();

                    cuda::count_init(device_id, p_out_valid, p_count, n_elem);
                }
                else
                {
                    const NT_MASK *p_out_valid = nullptr;

                    cuda::count_init(device_id, p_out_valid, p_count, n_elem);
                }
            }
            else
            {
#endif
                this->count = out_array->new_instance(n_elem);

                auto sp_count = static_cast<TT*>(this->count.get())->get_cpu_accessible();
                NT *p_count = sp_count.get();

                if (this->fill_value != -1)
                {
                    auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
                    const NT_MASK *p_out_valid = sp_out_valid.get();

                    for (unsigned int i = 0; i < n_elem; ++i)
                        p_count[i] = (p_out_valid[i]) ? 1. : 0.;
                }
                else
                {
                    for (unsigned int i = 0; i < n_elem; ++i)
                        p_count[i] = 1.;
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
        }
#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cuda_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cuda_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_count = static_cast<TT*>(this->count.get())->get_cuda_accessible();
            NT *p_count = sp_count.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cuda_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cuda_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cuda_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                cuda::average(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_count, p_red_array, p_red_valid, n_elem);
            }
            else
            {
                const NT_MASK *p_in_valid = nullptr;
                const NT_MASK *p_out_valid = nullptr;

                cuda::average(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_count, p_red_array, p_red_valid, n_elem);
            }
        }
        else
        {
#endif
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cpu_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cpu_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_count = static_cast<TT*>(this->count.get())->get_cpu_accessible();
            NT *p_count = sp_count.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cpu_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cpu_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cpu_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    //update the count only where there is valid data
                    if (p_in_valid[i])
                    {
                        p_count[i] += 1.;
                    }

                    // update the valid value mask
                    p_red_valid[i] = char(1);
                    if (p_in_valid[i] && p_out_valid[i])
                    {
                        // accumulate
                        p_red_array[i] = p_out_array[i] + p_in_array[i];
                    }
                    else if (p_in_valid[i] && !p_out_valid[i])
                    {
                        p_red_array[i] = p_in_array[i];
                    }
                    else if (!p_in_valid[i] && p_out_valid[i])
                    {
                        p_red_array[i] = p_out_array[i];
                    }
                    else
                    {
                        p_red_array[i] = this->fill_value;
                        p_red_valid[i] = char(0);
                    }
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    // update count
                    p_count[i] += 1.;

                    // update the valid value mask
                    p_red_valid[i] = char(1);

                    // accumulate
                    p_red_array[i] = p_out_array[i] + p_in_array[i];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
        return 0;
    )
}

// --------------------------------------------------------------------------
int summation_operator::update(
    int device_id,
    const const_p_teca_variant_array &out_array,
    const const_p_teca_variant_array &out_valid,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    p_teca_variant_array &red_array,
    p_teca_variant_array &red_valid)
{
    unsigned long n_elem = out_array->size();

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        // set the CUDA device to run on
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
        {
            TECA_ERROR("Failed to set the CUDA device to " << device_id
                << ". " << cudaGetErrorString(ierr))
            return -1;
        }

        red_array = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);
        red_valid = out_valid->new_instance(n_elem, teca_variant_array::allocator::cuda);
    }
    else
    {
#endif
        red_array = out_array->new_instance(n_elem);
        red_valid = out_valid->new_instance(n_elem);
#if defined(TECA_HAS_CUDA)
    }
#endif

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        red_array.get(),

        using NT_MASK = char;
        using TT_MASK = teca_variant_array_impl<NT_MASK>;

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cuda_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cuda_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cuda_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cuda_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cuda_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                cuda::summation(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
            else
            {
                const NT_MASK *p_in_valid = nullptr;
                const NT_MASK *p_out_valid = nullptr;

                cuda::summation(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
        }
        else
        {
#endif
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cpu_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cpu_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cpu_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cpu_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT*>(in_valid.get())->get_cpu_accessible();
                const NT *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT*>(out_valid.get())->get_cpu_accessible();
                const NT *p_out_valid = sp_out_valid.get();

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_red_valid[i] = char(1);
                    if (p_in_valid[i] && p_out_valid[i])
                    {
                        // accumulate
                        p_red_array[i] = p_out_array[i] + p_in_array[i];
                    }
                    else if (p_in_valid[i] && !p_out_valid[i])
                    {
                        p_red_array[i] = p_in_array[i];
                    }
                    else if (!p_in_valid[i] && p_out_valid[i])
                    {
                        p_red_array[i] = p_out_array[i];
                    }
                    else
                    {
                        p_red_valid[i] = char(0);
                    }
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_red_valid[i] = char(1);
                    p_red_array[i] = p_out_array[i] + p_in_array[i];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
        return 0;
    )
}

// --------------------------------------------------------------------------
int minimum_operator::update(
    int device_id,
    const const_p_teca_variant_array &out_array,
    const const_p_teca_variant_array &out_valid,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    p_teca_variant_array &red_array,
    p_teca_variant_array &red_valid)
{
    unsigned long n_elem = out_array->size();

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        // set the CUDA device to run on
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
        {
            TECA_ERROR("Failed to set the CUDA device to " << device_id
                << ". " << cudaGetErrorString(ierr))
            return -1;
        }

        red_array = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);
        red_valid = out_valid->new_instance(n_elem, teca_variant_array::allocator::cuda);
    }
    else
    {
#endif
        red_array = out_array->new_instance(n_elem);
        red_valid = out_valid->new_instance(n_elem);
#if defined(TECA_HAS_CUDA)
    }
#endif

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        red_array.get(),

        using NT_MASK = char;
        using TT_MASK = teca_variant_array_impl<NT_MASK>;

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cuda_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cuda_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cuda_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cuda_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cuda_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                cuda::minimum(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
            else
            {
                const NT_MASK *p_in_valid = nullptr;
                const NT_MASK *p_out_valid = nullptr;

                cuda::minimum(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
        }
        else
        {
#endif
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cpu_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cpu_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cpu_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cpu_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cpu_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    // update the valid value mask
                    p_red_valid[i] = char(1);
                    if (p_in_valid[i] && p_out_valid[i])
                        // reduce
                        p_red_array[i] = (p_in_array[i] < p_out_array[i]) ?
                                                        p_in_array[i] : p_out_array[i];
                    else if (p_in_valid[i] && !p_out_valid[i])
                    {
                        p_red_array[i] = p_in_array[i];
                    }
                    else if (!p_in_valid[i] && p_out_valid[i])
                    {
                        p_red_array[i] = p_out_array[i];
                    }
                    else
                    {
                        p_red_valid[i] = char(0);
                    }
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    // update the valid value mask
                    p_red_valid[i] = char(1);

                    // reduce
                    p_red_array[i] = p_in_array[i] < p_out_array[i] ?
                                                   p_in_array[i] : p_out_array[i];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
        return 0;
    )
}

// --------------------------------------------------------------------------
int maximum_operator::update(
    int device_id,
    const const_p_teca_variant_array &out_array,
    const const_p_teca_variant_array &out_valid,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    p_teca_variant_array &red_array,
    p_teca_variant_array &red_valid)
{
    unsigned long n_elem = out_array->size();

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        // set the CUDA device to run on
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
        {
            TECA_ERROR("Failed to set the CUDA device to " << device_id
                << ". " << cudaGetErrorString(ierr))
            return -1;
        }

        red_array = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);
        red_valid = out_valid->new_instance(n_elem, teca_variant_array::allocator::cuda);
    }
    else
    {
#endif
        red_array = out_array->new_instance(n_elem);
        red_valid = out_valid->new_instance(n_elem);
#if defined(TECA_HAS_CUDA)
    }
#endif

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        red_array.get(),

        using NT_MASK = char;
        using TT_MASK = teca_variant_array_impl<NT_MASK>;

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cuda_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cuda_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cuda_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cuda_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cuda_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                cuda::maximum(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
            else
            {
                const NT_MASK *p_in_valid = nullptr;
                const NT_MASK *p_out_valid = nullptr;

                cuda::maximum(device_id, p_out_array, p_out_valid,
                    p_in_array, p_in_valid, p_red_array, p_red_valid, n_elem);
            }
        }
        else
        {
#endif
            auto sp_in_array = static_cast<const TT*>(in_array.get())->get_cpu_accessible();
            const NT *p_in_array = sp_in_array.get();

            auto sp_out_array = static_cast<const TT*>(out_array.get())->get_cpu_accessible();
            const NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cpu_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_red_valid = static_cast<TT_MASK*>(red_valid.get())->get_cpu_accessible();
            NT_MASK *p_red_valid = sp_red_valid.get();

            // fix invalid values
            if (this->fill_value != -1)
            {
                auto sp_in_valid = static_cast<const TT_MASK*>(in_valid.get())->get_cpu_accessible();
                const NT_MASK *p_in_valid = sp_in_valid.get();

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    // update the valid value mask
                    p_red_valid[i] = char(1);
                    if (p_in_valid[i] && p_out_valid[i])
                        // reduce
                        p_red_array[i] = (p_in_array[i] > p_out_array[i]) ?
                                                        p_in_array[i] : p_out_array[i];
                    else if (p_in_valid[i] && !p_out_valid[i])
                    {
                        p_red_array[i] = p_in_array[i];
                    }
                    else if (!p_in_valid[i] && p_out_valid[i])
                    {
                        p_red_array[i] = p_out_array[i];
                    }
                    else
                    {
                        p_red_valid[i] = char(0);
                    }
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    // update the valid value mask
                    p_red_valid[i] = char(1);

                    // reduce
                    p_red_array[i] = p_in_array[i] > p_out_array[i] ?
                                                   p_in_array[i] : p_out_array[i];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
        return 0;
    )
}

// --------------------------------------------------------------------------
int reduction_operator_collection::finalize(
    int device_id,
    p_teca_variant_array &out_array,
    const p_teca_variant_array &out_valid,
    p_teca_variant_array &red_array)
{
    if (this->fill_value != -1)
    {
        TEMPLATE_DISPATCH_FP(
            teca_variant_array_impl,
            out_array.get(),

            unsigned long n_elem = out_array->size();

            auto sp_out_array = static_cast<TT*>(out_array.get())->get_cpu_accessible();
            NT *p_out_array = sp_out_array.get();

            using NT_MASK = char;
            using TT_MASK = teca_variant_array_impl<NT_MASK>;

            auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
            const NT_MASK *p_out_valid = sp_out_valid.get();

            NT fill_value = NT(this->fill_value);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                if (!p_out_valid[i])
                {
                    p_out_array[i] = fill_value;
                }
            }
            return 0;
        )
    }
}

// --------------------------------------------------------------------------
int average_operator::finalize(
    int device_id,
    p_teca_variant_array &out_array,
    const p_teca_variant_array &out_valid,
    p_teca_variant_array &red_array)
{
    unsigned long n_elem = out_array->size();

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        // set the CUDA device to run on
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaSetDevice(device_id)) != cudaSuccess)
        {
            TECA_ERROR("Failed to set the CUDA device to " << device_id
                << ". " << cudaGetErrorString(ierr))
            return -1;
        }

        red_array = out_array->new_instance(n_elem, teca_variant_array::allocator::cuda);
    }
    else
    {
#endif
        red_array = out_array->new_instance(n_elem);
#if defined(TECA_HAS_CUDA)
    }
#endif

    TEMPLATE_DISPATCH_FP(
        teca_variant_array_impl,
        red_array.get(),

        using NT_MASK = char;
        using TT_MASK = teca_variant_array_impl<NT_MASK>;

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto sp_out_array = static_cast<TT*>(out_array.get())->get_cuda_accessible();
            NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cuda_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_count = static_cast<const TT*>(this->count.get())->get_cuda_accessible();
            const NT *p_count = sp_count.get();

            if (this->fill_value != -1)
            {
                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cuda_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                cuda::finalize(device_id, p_out_array, p_out_valid,
                    p_red_array, p_count, this->fill_value, n_elem);
            }
            else
            {
                const NT_MASK *p_out_valid = nullptr;

                cuda::finalize(device_id, p_out_array, p_out_valid,
                    p_red_array, p_count, this->fill_value, n_elem);
            }
        }
        else
        {
#endif
            auto sp_out_array = static_cast<TT*>(out_array.get())->get_cpu_accessible();
            NT *p_out_array = sp_out_array.get();

            auto sp_red_array = static_cast<TT*>(red_array.get())->get_cpu_accessible();
            NT *p_red_array = sp_red_array.get();

            auto sp_count = static_cast<const TT*>(this->count.get())->get_cpu_accessible();
            const NT *p_count = sp_count.get();

            if (this->fill_value != -1)
            {
                // finish the average. We keep track of the invalid
                // values (these will have a zero count) set them to
                // the fill value

                auto sp_out_valid = static_cast<const TT_MASK*>(out_valid.get())->get_cpu_accessible();
                const NT_MASK *p_out_valid = sp_out_valid.get();

                NT fill_value = NT(this->fill_value);

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_red_array[i] = (!p_out_valid[i]) ?
                                      fill_value : p_out_array[i]/p_count[i];
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_red_array[i] = p_out_array[i]/p_count[i];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )
    this->count = nullptr;
    return 0;
}

// --------------------------------------------------------------------------
p_reduction_operator reduction_operator_factory::New(const std::string op)
{
    if (op == "summation")
    {
        return std::make_shared<summation_operator>();
    }
    else if (op == "average")
    {
        return std::make_shared<average_operator>();
    }
    else if (op == "minimum")
    {
        return std::make_shared<minimum_operator>();
    }
    else if (op == "maximum")
    {
        return std::make_shared<maximum_operator>();
    }
    TECA_FATAL_ERROR("Failed to construct a \""
        << op << "\" reduction operator")
    return nullptr;
}

// --------------------------------------------------------------------------
int teca_temporal_reduction::set_operator(const std::string &op)
{
    if (op == "average" ||
        op == "summation" ||
        op == "minimum" ||
        op == "maximum")
    {
        this->operator_name = op;
    }
    else
    {
        TECA_ERROR("Invalid operator name \"" << op << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_temporal_reduction::set_interval(const std::string &interval)
{
    if (interval == "daily" ||
        interval == "monthly" ||
        interval == "seasonal" ||
        interval == "yearly" ||
        interval.find("_step") != std::string::npos)
    {
        this->interval_name = interval;
    }
    else
    {
        TECA_ERROR("Invalid interval name \"" << interval << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_temporal_reduction::teca_temporal_reduction() :
    operator_name("None"), interval_name("None"), fill_value(-1), use_fill_value(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_temporal_reduction::~teca_temporal_reduction()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_temporal_reduction::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_temporal_reduction":prefix));

    opts.add_options()
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        TECA_POPTS_GET(int, prefix, use_fill_value,
            "controls how invalid or missing values are treated")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_temporal_reduction::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, double, prefix, fill_value)
    TECA_POPTS_SET(opts, int, prefix, use_fill_value)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_temporal_reduction::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &md_in)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_temporal_reduction::get_output_metadata" << std::endl;
#endif
    (void)port;

    // sanity checks
    if (this->interval_name == "None")
        TECA_FATAL_ERROR("No interval specified")

    if (this->operator_name == "None")
        TECA_FATAL_ERROR("No operator specified")

    if (this->point_arrays.empty())
        TECA_FATAL_ERROR("No arrays specified")

    teca_metadata md_out = md_in[0];

    //get the input time axis and metadata
    teca_metadata atts;
    md_out.get("attributes", atts);
    teca_metadata coords;
    md_out.get("coordinates", coords);

    const_p_teca_variant_array t = coords.get("t");
    std::string t_var;
    coords.get("t_variable", t_var);
    teca_metadata t_atts;
    atts.get(t_var, t_atts);

    std::string cal;
    try
    {
        t_atts.get("calendar", cal);
    }
    catch (...)
    {
        cal = "standard";
        TECA_WARNING("Attributes for the time axis " << t_var <<
            " is missing calendar. The \"standard\" calendar will be used")
    }

    std::string t_units;
    t_atts.get("units", t_units);

    teca_calendar_util::p_interval_iterator it =
        teca_calendar_util::interval_iterator_factory::New(this->interval_name);

    if (!it || it->initialize(t, t_units, cal, 0, -1))
    {
        TECA_ERROR("Failed to initialize the \""
            << this->interval_name << "\" iterator")
    }

    this->indices.clear();

    // convert the time axis to the specified interval
    while (*it)
    {
        teca_calendar_util::time_point first_step;
        teca_calendar_util::time_point last_step;

        it->get_next_interval(first_step, last_step);

        this->indices.push_back(
            time_interval(first_step.time, first_step.index, last_step.index));
    }

    size_t n_elem = this->indices.size();

    if (this->get_verbose() > 1)
    {
        std::cerr << teca_parallel_id()
        << "indices = [" << std::endl;
        for (size_t i = 0; i < n_elem; ++i)
            std::cerr << this->indices[i].time << " "
                      << this->indices[i].start_index << " "
                      << this->indices[i].end_index << std::endl;
        std::cerr << "]" << std::endl;
    }

    // update the pipeline control keys
    std::string initializer_key;
    md_out.get("index_initializer_key", initializer_key);
    long n_indices = this->indices.size();
    md_out.set(initializer_key, n_indices);

    // update the metadata so that modified time axis and reduced variables
    // are presented
    teca_metadata out_atts;
    std::set<std::string> out_vars;

    for (size_t i = 0; i < this->point_arrays.size(); ++i)
    {
        std::string &array = this->point_arrays[i];

        // name of the output array
        out_vars.insert(array);

        // pass the attributes
        teca_metadata in_atts;
        atts.get(array, in_atts);

        // convert integer to floating point for averaging operations
        if (this->operator_name == "average")
        {
            int tc;
            in_atts.get("type_code", tc);
            if (tc == ((int)teca_variant_array_code<int>::get()) ||
                tc == ((int)teca_variant_array_code<char>::get()) ||
                tc == ((int)teca_variant_array_code<short>::get()) ||
                tc == ((int)teca_variant_array_code<unsigned int>::get()) ||
                tc == ((int)teca_variant_array_code<unsigned char>::get()) ||
                tc == ((int)teca_variant_array_code<unsigned short>::get()))
            {
                tc = ((int)teca_variant_array_code<float>::get());
            }
            else if (
                tc == ((int)teca_variant_array_code<long>::get()) ||
                tc == ((int)teca_variant_array_code<long long>::get()) ||
                tc == ((int)teca_variant_array_code<unsigned long>::get()) ||
                tc == ((int)teca_variant_array_code<unsigned long long>::get()))
            {
                tc = ((int)teca_variant_array_code<double>::get());
            }
            in_atts.set("type_code", tc);
        }

        // document the transformation
        std::ostringstream oss;
        oss << this->interval_name << " "
            << this->operator_name << " of " << array;
        in_atts.set("description", oss.str());

        out_atts.set(array, in_atts);
    }

    // update time axis
    std::vector<double> t_out;
    for (size_t i = 0; i < n_elem; ++i)
    {
        t_out.push_back(this->indices[i].time);
    }
    coords.set("t", t_out);
    md_out.set("coordinates", coords);

    out_atts.set(t_var, t_atts);

    // package it all up and return
    md_out.set("variables", out_vars);
    md_out.set("attributes", out_atts);

    return md_out;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_temporal_reduction::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &md_in,
    const teca_metadata &req_in)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_temporal_reduction::get_upstream_request" << std::endl;
#endif
    (void)port;

    const teca_metadata md = md_in[0];

    // get the available arrays
    std::set<std::string> vars_in;
    if (md.has("variables"))
       md.get("variables", vars_in);

    // get the requested arrays
    std::set<std::string> req_arrays;
    if (req_in.has("arrays"))
        req_in.get("arrays", req_arrays);

    // get the array attributes
    teca_metadata atts;
    md.get("attributes", atts);

    for (size_t i = 0; i < this->point_arrays.size(); ++i)
    {
        std::string &array = this->point_arrays[i];

        // request the array
        if (!req_arrays.count(array))
            req_arrays.insert(array);

        double fill_value;
        std::string vv_mask = array + "_valid";
        if (this->use_fill_value &&
            vars_in.count(vv_mask) &&
            !req_arrays.count(vv_mask))
        {
            // request the associated valid value mask
            req_arrays.insert(vv_mask);

            // get the fill value
            teca_metadata array_atts;
            atts.get(array, array_atts);
            if (this->fill_value != -1)
            {
                fill_value = this->fill_value;
            }
            else if (array_atts.has("_FillValue"))
            {
                array_atts.get("_FillValue", fill_value);
            }
            else if (array_atts.has("missing_value"))
            {
                array_atts.get("missing_value", fill_value);
            }
        }

        // create and initialize the operator
        p_reduction_operator op
                      = reduction_operator_factory::New(this->operator_name);
        op->initialize(fill_value);

        // save the operator
        this->op[array] = op;
    }

    // generate one request for each time step in the interval
    std::vector<teca_metadata> up_reqs;
    std::string request_key;
    md.get("index_request_key", request_key);
    unsigned long req_id;
    req_in.get(request_key, req_id);
    int i = this->indices[req_id].start_index;
    while (i <= this->indices[req_id].end_index)
    {
        teca_metadata req(req_in);
        req.set("arrays", req_arrays);
        req.set(request_key, i);
        up_reqs.push_back(req);
        i += 1;
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_temporal_reduction::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &data_in,
    const teca_metadata &req_in,
    int streaming)
{
    (void)port;

    // get the requested ind
    std::string request_key;
    req_in.get("index_request_key", request_key);
    unsigned long req_id;
    req_in.get(request_key, req_id);

    int device_id = -1;
    req_in.get("device_id", device_id);

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_temporal_reduction::execute request "
        << req_id << " device " << device_id
        << " (" << this->indices[req_id].start_index
        << " - " << this->indices[req_id].end_index
        << "), reducing " << data_in.size() << ", "
        << streaming << " remain" << std::endl;
#endif

    size_t n_data = data_in.size();

    // copy the first mesh
    const_p_teca_cartesian_mesh mesh_in
      = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[n_data-1]);
    p_teca_cartesian_mesh mesh_out = teca_cartesian_mesh::New();
    mesh_out->shallow_copy(
      std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));
    p_teca_array_collection arrays_out = mesh_out->get_point_arrays();


    // accumulate incoming values
    for (int i = n_data-2; i >= 0; --i)
    {
        mesh_in
           = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[i]);
        const_p_teca_array_collection arrays_in = mesh_in->get_point_arrays();

        for (size_t j = 0; j < this->point_arrays.size(); ++j)
        {
            std::string &array = this->point_arrays[j];

            // the valid value masks
            std::string valid = array + "_valid";
            const_p_teca_variant_array in_valid = nullptr;
            const_p_teca_variant_array out_valid = nullptr;

            // arrays
            const_p_teca_variant_array in_array = arrays_in->get(array);
            const_p_teca_variant_array out_array = arrays_out->get(array);

            // valid value masks
            if (arrays_in->has(valid))
            {
               in_valid = arrays_in->get(valid);
            }
            if (arrays_out->has(valid))
            {
               out_valid = arrays_out->get(valid);
            }

            // apply the reduction
            p_teca_variant_array red_array;
            p_teca_variant_array red_valid = nullptr;
            this->op[array]->update(
                device_id, out_array, out_valid,
                in_array, in_valid, red_array, red_valid);

            // udpate the output
            arrays_out->set(array, red_array);

            if (red_valid != nullptr)
                arrays_out->set(valid, red_valid);
        }
    }

    // when all the data is processed
    if (!streaming)
    {
        for (size_t i = 0; i < this->point_arrays.size(); ++i)
        {
            std::string &array = this->point_arrays[i];

            // the valid value masks
            std::string valid = array + "_valid";
            p_teca_variant_array out_valid = nullptr;

            p_teca_variant_array out_array = arrays_out->get(array);
            if (arrays_out->has(valid))
               out_valid = arrays_out->get(valid);

            // finalize the reduction
            p_teca_variant_array red_array;
            this->op[array]->finalize(
                device_id, out_array, out_valid, red_array);

            // udpate the output
            if (this->operator_name == "average")
                arrays_out->set(array, red_array);
            else
                arrays_out->set(array, out_array);

            if (out_valid != nullptr)
               arrays_out->set(valid, out_valid);
        }

        // fix time
        mesh_out->set_time_step(req_id);
        mesh_out->set_time(this->indices[req_id].time);
    }

    return mesh_out;
}
