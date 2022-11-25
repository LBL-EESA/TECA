#include "teca_temporal_reduction.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_calendar_util.h"
#include "teca_valid_value_mask.h"

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

//#define TECA_DEBUG

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;



// PIMPL idiom hides internals
// defines the API for reduction operators
class teca_cpp_temporal_reduction::internals_t
{
public:
    internals_t() {}
    ~internals_t() {}

    /** check if the passed array contains integer data, if so deep-copy to
     * floating point type. 32(64) bit integers will be copied to 32(64) bit
     * floating point.
     * @param[in] alloc the allocator to use for the new array if a deep-copy is made
     * @param[inout] array the array to check and convert from integer to floating point
     */
    static
    void ensure_floating_point(allocator alloc, const_p_teca_variant_array &array);

public:
    class reduction_operator;
    class average_operator;
    class summation_operator;
    class minimum_operator;
    class maximum_operator;
    class reduction_operator_factory;
    class time_interval;

    using p_reduction_operator = std::shared_ptr<reduction_operator>;

public:
    std::vector<time_interval> indices;
    std::map<std::string, p_reduction_operator> op;
};

// --------------------------------------------------------------------------
void teca_cpp_temporal_reduction::internals_t::ensure_floating_point(
     allocator alloc, const_p_teca_variant_array &array)
{
    if (std::dynamic_pointer_cast<const teca_variant_array_impl<double>>(array)  ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<float>>(array))
    {
        // the data is already floating point type
        return;
    }
    else if (std::dynamic_pointer_cast<const teca_variant_array_impl<long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<long long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<unsigned long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<unsigned long long>>(array))
    {
        // convert from a 64 bit integer to a 64 bit floating point
        size_t n_elem = array->size();
        p_teca_double_array tmp = teca_double_array::New(n_elem, alloc);
        tmp->set(0, array, 0, n_elem);
        array = tmp;
    }
    else
    {
        // convert from a 32 bit integer to a 32 bit floating point
        size_t n_elem = array->size();
        p_teca_float_array tmp = teca_float_array::New(n_elem, alloc);
        tmp->set(0, array, 0, n_elem);
        array = tmp;
    }
}


/** defines the API for operators implementing reduction calculations.
 *
 * implements 4 class methods: initialize, update_cpu, update_gpu, and
 * finalize.
 *
 *  initialize:
 *      initializes the reduction. If a fill value is passed
 *      the operator should use it to identify missing values in the
 *      data and handle them appropriately.
 *  update:
 *      reduces arrays_in (new data) into arrays_out (current state) and
 *      returns the result. For update_cpu, the target device is cpu;
 *      and for update_gpu, the target device is gpu (cuda).
 *  finalize:
 *      finalizes arrays_out (current state) and returns the result.
 *      If no finalization is needed simply return arrays_out.
 */
class teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:

    reduction_operator() : fill_value(-1) {}

    virtual void initialize(double fill_value);

    virtual int update_cpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) = 0;

    virtual int update_gpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) = 0;

    virtual int finalize(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out);

public:
    double fill_value;
};


/// implements time average
class teca_cpp_temporal_reduction::internals_t::average_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:

    int update_cpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

    int update_gpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

    int finalize(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out) override;
};


/// implements sum over time
class teca_cpp_temporal_reduction::internals_t::summation_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int update_cpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

    int update_gpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

};


/// implements minimum over time
class teca_cpp_temporal_reduction::internals_t::minimum_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int update_cpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

    int update_gpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;
};


/// implements maximum over time
class teca_cpp_temporal_reduction::internals_t::maximum_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int update_cpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;

    int update_gpu(int device_id,
          const std::string &array,
          p_teca_array_collection &arrays_out,
          p_teca_array_collection &arrays_in) override;
};


/// constructs reduction_operator
class teca_cpp_temporal_reduction::internals_t::reduction_operator_factory
{
public:
    /** Allocate and return an instance of the named operator
     * @param[in] op Id of the desired reduction operator.
     *               One of average, summation, minimum, or
     *                                              maximum
     * @returns an instance of reduction_operator
     */
    static teca_cpp_temporal_reduction::internals_t::p_reduction_operator New(
                                                                   int op);
};


/// A time interval defined by a sart and end time step
struct teca_cpp_temporal_reduction::internals_t::time_interval
{
    time_interval(double t, long start, long end) : time(t),
        start_index(start), end_index(end)
    {}

    double time;
    long start_index;
    long end_index;
};


// --------------------------------------------------------------------------
void teca_cpp_temporal_reduction::internals_t::reduction_operator::initialize(
     double fill_value)
{
    this->fill_value = fill_value;
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

    p_count[i] = (p_out_valid[i]) ? 1. : 0.;
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
     T *p_count, T *p_red_count,
     T *p_red_array, char *p_red_valid,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_out_valid == nullptr)
    {
        // accumulate
        p_red_array[i] = p_out_array[i] + p_in_array[i];
    }
    else
    {
        p_red_valid[i] = char(1);
        //update the count only where there is valid data
        if (p_in_valid[i])
        {
            p_red_count[i] = p_count[i] + 1.;
        }
        else
        {
            p_red_count[i] = p_count[i];
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
    T *p_count, T *p_red_count,
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
         p_in_array, p_in_valid, p_count, p_red_count, p_red_array, p_red_valid, n_elem);
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

    if (p_out_valid == nullptr)
    {
        // accumulate
        p_red_array[i] = p_out_array[i] + p_in_array[i];
    }
    else
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

    if (p_out_valid == nullptr)
    {
        // reduce
        p_red_array[i] = (p_in_array[i] < p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
    }
    else
    {
        p_red_valid[i] = char(1);
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

    if (p_out_valid == nullptr)
    {
        // reduce
        p_red_array[i] = (p_in_array[i] > p_out_array[i]) ?
                                        p_in_array[i] : p_out_array[i];
    }
    else
    {
        p_red_valid[i] = char(1);
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
     double fill_value,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (!p_out_valid[i])
    {
        p_out_array[i] = fill_value;
    }
}

// --------------------------------------------------------------------------
template <typename T>
int finalize(
    int device_id,
    T *p_out_array, const char *p_out_valid,
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
          fill_value, n_elem);
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
void average_finalize(
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
        p_red_array[i] = p_out_array[i]/p_count[0];
    }
    else
    {
        p_red_array[i] = (!p_out_valid[i]) ?
                                 fill_value : p_out_array[i]/p_count[i];
    }
}

// --------------------------------------------------------------------------
template <typename T>
int average_finalize(
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
    average_finalize<<<block_grid,thread_grid>>>(p_out_array, p_out_valid,
          p_red_array, p_count, fill_value, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the finalize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}
} //end of namespace cuda
#endif

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::update_cpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
    (void)device_id;

    // get the input and output arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // don't use integer types for this calculation
    internals_t::ensure_floating_point(allocator::malloc, in_array);
    internals_t::ensure_floating_point(allocator::malloc, out_array);

    // get the valid value masks
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // get the current count
    size_t n_elem_count = out_valid ? n_elem : 1;
    std::string count_name = array + "_count";
    p_teca_variant_array count = arrays_out->get(count_name);
    if (!count)
    {
        // allocate and initialize the count
        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_array.get(),
            auto [tmp, p_tmp] = ::New<TT>(n_elem_count);
            if (out_valid)
            {
                auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);

                for (unsigned int i = 0; i < n_elem; ++i)
                    p_tmp[i] = (p_out_valid[i]) ? NT(1) : NT(0);
            }
            else
            {
                p_tmp[0] = 1.;
            }
            count = tmp;
        )
    }

    // the result of this pass
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;
    p_teca_variant_array red_count;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem);
        auto [res_count, p_res_count] = ::New<TT>(n_elem_count);

        auto [sp_in_array, p_in_array] = get_cpu_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cpu_accessible<CTT>(out_array);

        auto [p_count] = data<TT>(count);

        if (out_valid)
        {
            // update, respecting missing data

            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem);

            auto [sp_in_valid, p_in_valid] = get_cpu_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                // count where there is valid data, otherwise pass the current count through
                if (p_in_valid[i])
                {
                    p_res_count[i] = p_count[i] + NT(1);
                }
                else
                {
                    p_res_count[i] = p_count[i];
                }

                // update, respecting missing data
                p_res_valid[i] = NT_MASK(1);
                if (p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i] + p_in_array[i];
                }
                else if (p_in_valid[i] && !p_out_valid[i])
                {
                    p_res_array[i] = p_in_array[i];
                }
                else if (!p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i];
                }
                else
                {
                    p_res_array[i] = NT(this->fill_value);
                    p_res_valid[i] = NT_MASK(0);
                }
            }

            red_valid = res_valid;
        }
        else
        {
            // update , no misisng data
            p_res_count[0] = p_count[0] + NT(1);

            for (unsigned int i = 0; i < n_elem; ++i)
                p_res_array[i] = p_out_array[i] + p_in_array[i];
        }

        red_array = res_array;
        red_count = res_count;
    )

    // update the output
    arrays_out->set(array, red_array);
    arrays_out->set(count_name, red_count);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::update_gpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
#if defined(TECA_HAS_CUDA)
    // get the input and output arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // don't use integer types for this calculation
    internals_t::ensure_floating_point(allocator::cuda_async, in_array);
    internals_t::ensure_floating_point(allocator::cuda_async, out_array);

    // get the valid value masks
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // allocate space for the output mask which is a function of the current
    // input and previous state and size and place the count array
    size_t n_elem_count = 1;
    allocator alloc_count = allocator::malloc;
    if (out_valid)
    {
        // when we have the valid value mask we need a per-mesh element count
        n_elem_count = n_elem;
        alloc_count = allocator::cuda_async;
    }

    // get the current count
    std::string count_name = array + "_count";
    p_teca_variant_array count = arrays_out->get(count_name);
    if (!count)
    {
        // allocate and initialize the count
        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_array.get(),
            auto [tmp, p_tmp] = ::New<TT>(n_elem_count, alloc_count);
            if (out_valid)
            {
                auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);
                cuda::count_init(device_id, p_out_valid, p_tmp, n_elem_count);
            }
            else
            {
                p_tmp[0] = NT(1);
            }
            count = tmp;
        )
    }

    // results of the calc
    p_teca_variant_array red_array;
    p_teca_variant_array red_count;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem, allocator::cuda_async);
        auto [res_count, p_res_count] = ::New<TT>(n_elem_count, alloc_count);

        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cuda_accessible<CTT>(out_array);

        auto [p_count] = data<TT>(count);

        if (out_valid)
        {
            // update while respecting invalid/missing data
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem, allocator::cuda_async);

            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);

            cuda::average(device_id, p_out_array, p_out_valid,
                p_in_array, p_in_valid, p_count, p_res_count,
                p_res_array, p_res_valid, n_elem);

            red_valid = res_valid;
        }
        else
        {
            // update, no missing data
            p_res_count[0] = p_count[0] + 1.;

            cuda::summation(device_id, p_out_array, nullptr,
                p_in_array, nullptr, p_res_array, nullptr,
                n_elem);
        }

        red_array = res_array;
        red_count = res_count;
    )

    // update the output
    arrays_out->set(array, red_array);
    arrays_out->set(count_name, red_count);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
#else
    (void)device_id;
    (void)array;
    (void)arrays_out;
    (void)arrays_in;

    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::summation_operator::update_cpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
    (void)device_id;

    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // update reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [sp_in_array, p_in_array] = get_cpu_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cpu_accessible<CTT>(out_array);

        auto [res_array, p_res_array] = ::New<TT>(n_elem);

        if (out_valid)
        {
            // update, respect missing values
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem);

            auto [sp_in_valid, p_in_valid] = get_cpu_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_valid[i] = NT_MASK(1);
                if (p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i] + p_in_array[i];
                }
                else if (p_in_valid[i] && !p_out_valid[i])
                {
                    p_res_array[i] = p_in_array[i];
                }
                else if (!p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i];
                }
                else
                {
                    p_res_valid[i] = NT_MASK(0);
                }
            }

            red_valid = res_valid;
        }
        else
        {
            // update, no misisng values
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_array[i] = p_out_array[i] + p_in_array[i];
            }
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::summation_operator::update_gpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
#if defined(TECA_HAS_CUDA)
    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // partial update the reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem, allocator::cuda_async);

        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cuda_accessible<CTT>(out_array);

        if (out_valid)
        {
            // update, respect missing values
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem, allocator::cuda_async);

            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);

            cuda::summation(device_id, p_out_array, p_out_valid,
                p_in_array, p_in_valid, p_res_array, p_res_valid, n_elem);

            red_valid = res_valid;
        }
        else
        {
            cuda::summation(device_id, p_out_array, nullptr,
                p_in_array, nullptr, p_res_array, nullptr, n_elem);
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
#else
    (void)device_id;
    (void)array;
    (void)arrays_out;
    (void)arrays_in;

    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::minimum_operator::update_cpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
    (void)device_id;

    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // partial update the reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem);

        auto [sp_in_array, p_in_array] = get_cpu_accessible<TT>(in_array);
        auto [sp_out_array, p_out_array] = get_cpu_accessible<TT>(out_array);

        if (out_valid)
        {
            // update, respect missing values
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem);

            auto [sp_in_valid, p_in_valid] = get_cpu_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_valid[i] = NT_MASK(1);
                if (p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = (p_in_array[i] < p_out_array[i]) ?
                                                    p_in_array[i] : p_out_array[i];
                }
                else if (p_in_valid[i] && !p_out_valid[i])
                {
                    p_res_array[i] = p_in_array[i];
                }
                else if (!p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i];
                }
                else
                {
                    p_res_valid[i] = NT_MASK(0);
                }
            }

            red_valid = res_valid;
        }
        else
        {
            // update, no missing values
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_array[i] = p_in_array[i] < p_out_array[i] ?
                                               p_in_array[i] : p_out_array[i];
            }
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::minimum_operator::update_gpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
#if defined(TECA_HAS_CUDA)
    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // partial update of the reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem, allocator::cuda_async);
        auto [sp_in_array, p_in_array] = get_cuda_accessible<TT>(in_array);
        auto [sp_out_array, p_out_array] = get_cuda_accessible<TT>(out_array);

        if (out_valid)
        {
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem, allocator::cuda_async);
            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);

            cuda::minimum(device_id, p_out_array, p_out_valid,
                p_in_array, p_in_valid, p_res_array, p_res_valid, n_elem);

            red_valid = res_valid;
        }
        else
        {
            cuda::minimum(device_id, p_out_array, nullptr,
                p_in_array, nullptr, p_res_array, nullptr, n_elem);
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (red_valid != nullptr)
        arrays_out->set(valid, red_valid);

    return 0;
#else
    (void)device_id;
    (void)array;
    (void)arrays_out;
    (void)arrays_in;

    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::maximum_operator::update_cpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
    (void)device_id;

    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // partial update of the reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem);

        auto [sp_in_array, p_in_array] = get_cpu_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cpu_accessible<CTT>(out_array);

        if (out_valid)
        {
            // update, respect missing values
            auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem);

            auto [sp_in_valid, p_in_valid] = get_cpu_accessible<CTT_MASK>(in_valid);
            auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_valid[i] = NT_MASK(1);
                if (p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = (p_in_array[i] > p_out_array[i]) ?
                                                    p_in_array[i] : p_out_array[i];
                }
                else if (p_in_valid[i] && !p_out_valid[i])
                {
                    p_res_array[i] = p_in_array[i];
                }
                else if (!p_in_valid[i] && p_out_valid[i])
                {
                    p_res_array[i] = p_out_array[i];
                }
                else
                {
                    p_res_valid[i] = NT_MASK(0);
                }
            }

            red_valid = res_valid;
        }
        else
        {
            // update, no missing values
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                p_res_array[i] = p_in_array[i] > p_out_array[i] ?
                                               p_in_array[i] : p_out_array[i];
            }
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::maximum_operator::update_gpu(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out,
    p_teca_array_collection &arrays_in)
{
#if defined(TECA_HAS_CUDA)
    // arrays
    const_p_teca_variant_array in_array = arrays_in->get(array);
    const_p_teca_variant_array out_array = arrays_out->get(array);

    unsigned long n_elem = out_array->size();

    // the valid value masks
    // returns a nullptr if the array is not in the collection
    std::string valid = array + "_valid";
    const_p_teca_variant_array in_valid = arrays_in->get(valid);
    const_p_teca_variant_array out_valid = arrays_out->get(valid);

    // partial update of the reduction
    p_teca_variant_array red_array;
    p_teca_variant_array red_valid;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        auto [res_array, p_res_array] = ::New<TT>(n_elem, allocator::cuda_async);
        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [sp_out_array, p_out_array] = get_cuda_accessible<CTT>(out_array);

        if (out_valid)
        {
            // update, respect missing values
           auto [res_valid, p_res_valid] = ::New<TT_MASK>(n_elem, allocator::cuda_async);
           auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
           auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);

            cuda::maximum(device_id, p_out_array, p_out_valid,
                p_in_array, p_in_valid, p_res_array, p_res_valid, n_elem);

            red_valid = res_valid;
        }
        else
        {
            // update, no misisng values
            cuda::maximum(device_id, p_out_array, nullptr,
                p_in_array, nullptr, p_res_array, nullptr, n_elem);
        }

        red_array = res_array;
    )

    // update the output
    arrays_out->set(array, red_array);

    if (out_valid)
        arrays_out->set(valid, red_valid);

    return 0;
#else
    (void)device_id;
    (void)array;
    (void)arrays_out;
    (void)arrays_in;

    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::reduction_operator::finalize(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    p_teca_variant_array out_array = arrays_out->get(array);

    // the valid value masks
    std::string valid = array + "_valid";
    p_teca_variant_array out_valid = arrays_out->get(valid);

    if (out_valid)
    {
        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_array.get(),

            NT fill_value = NT(this->fill_value);
            unsigned long n_elem = out_array->size();

            auto [p_out_array] = data<TT>(out_array);

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);
                cuda::finalize(device_id, p_out_array, p_out_valid, fill_value, n_elem);
            }
            else
            {
#endif
                auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_out_array[i] = p_out_valid[i] ? p_out_array[i] : fill_value;
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
        // update the output
        arrays_out->set(array, out_array);
        )
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::finalize(
    int device_id,
    const std::string &array,
    p_teca_array_collection &arrays_out)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    p_teca_variant_array out_array = arrays_out->get(array);

    // get the valid value masks
    std::string valid = array + "_valid";
    p_teca_variant_array out_valid = arrays_out->get(valid);

    // get the count, always have it, but it will be per point when dealing
    // with valid value masks
    p_teca_variant_array count = arrays_out->get(array + "_count");

    // finalize the reduciton by scaling by 1 / count
                // finish the average. We keep track of the invalid
                // values (these will have a zero count) set them to
                // the fill value
    unsigned long n_elem = out_array->size();

    p_teca_variant_array red_array;

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        out_array.get(),

        NT fill_val = NT(this->fill_value);

        auto [p_out_array] = data<TT>(out_array);
        auto [p_count] = data<TT>(count);

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            // GPU
            auto [res_array, p_res_array] = ::New<TT>(n_elem, allocator::cuda_async);
            if (out_valid)
            {
                auto [sp_out_valid, p_out_valid] = get_cuda_accessible<CTT_MASK>(out_valid);

                cuda::average_finalize(device_id, p_out_array, p_out_valid,
                    p_res_array, p_count, fill_val, n_elem);
            }
            else
            {
                cuda::average_finalize(device_id, p_out_array, nullptr,
                    p_res_array, p_count, fill_val, n_elem);
            }
            red_array = res_array;
        }
        else
        {
#endif
            // CPU
            auto [res_array, p_res_array] = ::New<TT>(n_elem);
            if (out_valid)
            {
                auto [sp_out_valid, p_out_valid] = get_cpu_accessible<CTT_MASK>(out_valid);
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_res_array[i] = (!p_out_valid[i]) ?
                                      fill_val : p_out_array[i]/p_count[i];
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_res_array[i] = p_out_array[i]/p_count[0];
                }
            }

            red_array = res_array;
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

    // update the output
    arrays_out->set(array, red_array);

    return 0;
}

// --------------------------------------------------------------------------
teca_cpp_temporal_reduction::internals_t::p_reduction_operator
    teca_cpp_temporal_reduction::internals_t::reduction_operator_factory::New(
    int op)
{
    if (op == summation)
    {
        return std::make_shared<summation_operator>();
    }
    else if (op == average)
    {
        return std::make_shared<average_operator>();
    }
    else if (op == minimum)
    {
        return std::make_shared<minimum_operator>();
    }
    else if (op == maximum)
    {
        return std::make_shared<maximum_operator>();
    }
    TECA_ERROR("Failed to construct a \""
        << op << "\" reduction operator")
    return nullptr;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::set_operator(const std::string &op)
{
    if (op == "average")
    {
        this->op = average;
    }
    else if (op == "summation")
    {
        this->op = summation;
    }
    else if (op == "minimum")
    {
        this->op = minimum;
    }
    else if (op == "maximum")
    {
        this->op = maximum;
    }
    else
    {
        TECA_ERROR("Invalid operator name \"" << op << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::string teca_cpp_temporal_reduction::get_operator_name()
{
    std::string name;
    switch(this->op)
    {
        case average:
            name = "average";
            break;
        case summation:
            name = "summation";
            break;
        case minimum:
            name = "minimum";
            break;
        case maximum:
            name = "maximum";
            break;
        default:
            TECA_ERROR("Invalid \"operator\" " << this->op)
    }
    return name;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::set_interval(const std::string &interval)
{
    if (interval == "daily")
    {
        this->interval = daily;
    }
    else if (interval == "monthly")
    {
        this->interval = monthly;
    }
    else if (interval == "seasonal")
    {
        this->interval = seasonal;
    }
    else if (interval == "yearly")
    {
        this->interval = yearly;
    }
    else if (interval == "n_steps")
    {
        this->interval = n_steps;
    }
    else if (interval == "all")
    {
        this->interval = all;
    }
    else
    {
        TECA_ERROR("Invalid interval name \"" << interval << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::string teca_cpp_temporal_reduction::get_interval_name()
{
    std::string name;
    switch(this->interval)
    {
        case daily:
            name = "daily";
            break;
        case monthly:
            name = "monthly";
            break;
        case seasonal:
            name = "seasonal";
            break;
        case yearly:
            name = "yearly";
            break;
        case n_steps:
            name = "n_steps";
            break;
        case all:
            name = "all";
            break;
        default:
            TECA_ERROR("Invalid \"interval\" " << this->interval)
    }
    return name;
}

// --------------------------------------------------------------------------
teca_cpp_temporal_reduction::teca_cpp_temporal_reduction() :
    op(average), interval(monthly), number_of_steps(0), fill_value(-1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->set_stream_size(2);

    this->internals = new teca_cpp_temporal_reduction::internals_t;
}

// --------------------------------------------------------------------------
teca_cpp_temporal_reduction::~teca_cpp_temporal_reduction()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cpp_temporal_reduction::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cpp_temporal_reduction":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::vector<std::string>, prefix, point_arrays,
            "list of point centered arrays to process")
        TECA_POPTS_GET(int, prefix, op,
            "reduction operator to use"
            " (summation, minimum, maximum, or average)")
        TECA_POPTS_GET(int, prefix, interval,
            "interval to reduce the time axis to"
            " (daily, monthly, seasonal, yearly, n_steps, all)")
        TECA_POPTS_GET(long, prefix, number_of_steps,
            "desired number of steps for the n_steps interval")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        ;

    this->teca_threaded_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cpp_temporal_reduction::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_threaded_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, point_arrays)
    TECA_POPTS_SET(opts, int, prefix, op)
    TECA_POPTS_SET(opts, int, prefix, interval)
    TECA_POPTS_SET(opts, long, prefix, number_of_steps)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_cpp_temporal_reduction::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &md_in)
{
    if (this->get_verbose() > 1)
    {
        std::cerr << teca_parallel_id()
            << "teca_cpp_temporal_reduction::get_output_metadata" << std::endl;
    }

    (void)port;

    // sanity checks
    if (this->point_arrays.empty())
    {
        TECA_ERROR("No arrays specified")
        return teca_metadata();
    }

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
    if (t_atts.get("calendar", cal))
    {
        cal = "standard";
        TECA_WARNING("Attributes for the time axis " << t_var <<
            " is missing calendar. The \"standard\" calendar will be used")
    }

    std::string t_units;
    if (t_atts.get("units", t_units))
    {
        TECA_ERROR("Failed to get units")
        return teca_metadata();
    }

    teca_calendar_util::p_interval_iterator it;
    it = teca_calendar_util::interval_iterator_factory::New(
                                            this->interval);

    if (!it)
    {
        TECA_ERROR("Failed to allocate an instance of the \""
            << this->interval << "\" iterator")
        return teca_metadata();
    }

    if (this->interval == n_steps)
    {
        std::shared_ptr<teca_calendar_util::n_steps_iterator> itd
          = std::dynamic_pointer_cast<teca_calendar_util::n_steps_iterator>(it);
        if (itd)
        {
            itd->set_number_of_steps(this->number_of_steps);
        }
        else
        {
            TECA_ERROR("Failed to set the number of steps")
            return teca_metadata();
        }
    }

    if (it->initialize(t, t_units, cal, 0, -1))
    {
        TECA_ERROR("Failed to initialize the \""
            << this->interval << "\" iterator")
        return teca_metadata();
    }

    this->internals->indices.clear();

    // convert the time axis to the specified interval
    while (*it)
    {
        teca_calendar_util::time_point first_step;
        teca_calendar_util::time_point last_step;

        it->get_next_interval(first_step, last_step);

        this->internals->indices.push_back(
            teca_cpp_temporal_reduction::internals_t::time_interval(
            first_step.time, first_step.index, last_step.index));
    }

    long n_indices = this->internals->indices.size();

    if (this->get_verbose() > 1)
    {
        std::cerr << teca_parallel_id()
        << "indices = [" << std::endl;
        for (long i = 0; i < n_indices; ++i)
            std::cerr << this->internals->indices[i].time << " "
                      << this->internals->indices[i].start_index << " "
                      << this->internals->indices[i].end_index << std::endl;
        std::cerr << "]" << std::endl;
    }

    // update the pipeline control keys
    std::string initializer_key;
    md_out.get("index_initializer_key", initializer_key);
    md_out.set(initializer_key, n_indices);

    // update the metadata so that modified time axis and reduced variables
    // are presented
    teca_metadata out_atts;
    std::set<std::string> out_vars;

    size_t n_array = this->point_arrays.size();

    for (size_t i = 0; i < n_array; ++i)
    {
        const std::string &array = this->point_arrays[i];

        // name of the output array
        out_vars.insert(array);

        // pass the attributes
        teca_metadata in_atts;
        if (atts.get(array, in_atts))
        {
            TECA_ERROR("Failed to get the attributes for \""
                << array << "\"")
            return teca_metadata();
        }

        // convert integer to floating point for averaging operations
        if (this->op == average)
        {
            int tc;
            if (in_atts.get("type_code", tc))
            {
                TECA_ERROR("Failed to get type_code")
                return teca_metadata();
            }

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
        oss << this->get_interval_name() << " "
            << this->get_operator_name() << " of " << array;
        in_atts.set("description", oss.str());

        out_atts.set(array, in_atts);
    }

    // update time axis
    std::vector<double> t_out;
    for (long i = 0; i < n_indices; ++i)
    {
        t_out.push_back(this->internals->indices[i].time);
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
std::vector<teca_metadata> teca_cpp_temporal_reduction::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &md_in,
    const teca_metadata &req_in)
{
    if (this->get_verbose() > 1)
    {
        std::cerr << teca_parallel_id()
            << "teca_cpp_temporal_reduction::get_upstream_request" << std::endl;
    }

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

    size_t n_array = this->point_arrays.size();

    for (size_t i = 0; i < n_array; ++i)
    {
        const std::string &array = this->point_arrays[i];

        // request the array
        if (!req_arrays.count(array))
            req_arrays.insert(array);

        double fill_value = -1;
        std::string vv_mask = array + "_valid";
        if (vars_in.count(vv_mask) &&
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
        teca_cpp_temporal_reduction::internals_t::p_reduction_operator op
             = teca_cpp_temporal_reduction::internals_t::reduction_operator_factory::New(
                                                                 this->op);
        op->initialize(fill_value);

        // save the operator
        this->internals->op[array] = op;
    }

    // generate one request for each time step in the interval
    std::vector<teca_metadata> up_reqs;

    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_ERROR("Failed to locate the index_request_key")
        return up_reqs;
    }

    unsigned long req_id[2];
    if (req_in.get(request_key, req_id))
    {
        TECA_ERROR("Failed to get the requested index using the"
            " index_request_key \"" << request_key << "\"")
        return up_reqs;
    }

    int i = this->internals->indices[req_id[0]].start_index;
    while (i <= this->internals->indices[req_id[0]].end_index)
    {
        teca_metadata req(req_in);
        req.set("arrays", req_arrays);
        req.set(request_key, {i, i});
        up_reqs.push_back(req);
        i += 1;
    }

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cpp_temporal_reduction::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &data_in,
    const teca_metadata &req_in,
    int streaming)
{
    (void)port;

    // get the requested ind
    unsigned long req_id[2] = {0};
    std::string request_key;

    if (req_in.get("index_request_key", request_key) ||
        req_in.get(request_key, req_id))
    {
        TECA_ERROR("metadata issue. failed to get the requested indices")
        return nullptr;
    }

    // get the assigned GPU or CPU
    int device_id = -1;
    req_in.get("device_id", device_id);

    if (this->get_verbose())
    {
        std::cerr << teca_parallel_id()
            << "teca_cpp_temporal_reduction::execute request "
            << req_id[0] << " device " << device_id
            << " (" << this->internals->indices[req_id[0]].start_index
            << " - " << this->internals->indices[req_id[0]].end_index
            << "), reducing " << data_in.size() << ", "
            << streaming << " remain" << std::endl;
    }

    allocator alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
       alloc = allocator::cuda_async;

       if (teca_cuda_util::set_device(device_id))
           return nullptr;
    }
#endif

    long n_data = data_in.size();

    // We are processing data_in in reverse order because of the average
    // operator, more precisely because of its count variable.
    //
    // Note: the dependence on the last input dataset containing the previous result
    // is a private implementation detail of the streaming implementation and
    // should not be relied upon. The reason we currently do it this is that
    // we need per-thread cache and the output dataset does the trick. To access
    // this cache we need to know which of the incoming datasets is the
    // intermediate result.

    // get the last mesh, when streaming the will hold the previous results.
    p_teca_cartesian_mesh mesh_in =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(data_in[n_data-1]));

    p_teca_array_collection arrays_in;

    // make a shallow copy to hold the results of this pass
    p_teca_cartesian_mesh mesh_out = teca_cartesian_mesh::New();

    mesh_out->shallow_copy(
      std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));

    p_teca_array_collection arrays_out = mesh_out->get_point_arrays();

    size_t n_array = this->point_arrays.size();

    // Handle the case where the number of inputs < 2. Another intended private
    // implementation detail of teca's streaming is that the only time the
    // number of inputs will be less than 2 is when the current input interval
    // has only 1 time step. Hence in teca's current implementation this is an
    // edge case. In this case we'll set the result to the _FillValue. This
    // output is invalid and should not be used.
    if (n_data < 2)
    {
        arrays_in = mesh_in->get_point_arrays();

        for (size_t j = 0; j < n_array; ++j)
        {
            const std::string &array = this->point_arrays[j];

            // get the input array
            const_p_teca_variant_array in_array = arrays_in->get(array);
            unsigned long n_elem = in_array->size();

            bool have_vv_mask = arrays_in->has(array + "_valid");

            TEMPLATE_DISPATCH(
                teca_variant_array_impl,
                in_array.get(),

                if (this->op == average)
                {
                    // the average requires a floating point type due to truncation
                    // that occurs with integer division.
                    internals_t::ensure_floating_point(alloc, in_array);

                    // the average requires the count of valid values at each
                    // grid point (or mesh wide) is used.
                    size_t n_elem_count = 1;
                    allocator alloc_count = allocator::malloc;
                    if (have_vv_mask)
                    {
                        n_elem_count = n_elem;
                        alloc_count = (device_id >= 0 ? allocator::cuda_async : allocator::malloc);
                    }

                    p_teca_variant_array_impl<NT> count =
                        teca_variant_array_impl<NT>::New(n_elem_count, NT(1), alloc_count);

                    arrays_out->set(array + "_count", count);
                }

                // this output is invalid so mark all points with the _FillValue
                /*p_teca_variant_array_impl<NT> out_array =
                    teca_variant_array_impl<NT>::New(n_elem, NT(this->fill_value), alloc);
                arrays_out->set(array, out_array);*/

                p_teca_variant_array_impl<NT_MASK> out_mask =
                    teca_variant_array_impl<NT_MASK>::New(n_elem, NT_MASK(0), alloc);

                arrays_out->set(array + "_valid", out_mask);
                )
        }
    }

    // accumulate incoming values, in reverse order.
    for (long i = n_data - 2; i >= 0; --i)
    {
        mesh_in = std::dynamic_pointer_cast<teca_cartesian_mesh>(
          std::const_pointer_cast<teca_dataset>(data_in[i]));

        arrays_in = mesh_in->get_point_arrays();

        for (size_t j = 0; j < n_array; ++j)
        {
            const std::string &array = this->point_arrays[j];

            if (!(arrays_in->has(array)) || !(arrays_out->has(array)))
            {
                TECA_ERROR("array \"" << array << "\" not found")
                return nullptr;
            }

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                // apply the reduction
                this->internals->op[array]->update_gpu(
                    device_id, array, arrays_out, arrays_in);
            }
            else
            {
#endif
                // apply the reduction
                this->internals->op[array]->update_cpu(
                    device_id, array, arrays_out, arrays_in);
#if defined(TECA_HAS_CUDA)
            }
#endif
        }
    }

    // when all the data is processed
    if (!streaming)
    {
        for (size_t i = 0; i < n_array; ++i)
        {
            const std::string &array = this->point_arrays[i];

            // finalize the reduction
            this->internals->op[array]->finalize(
                device_id, array, arrays_out);
       }

        // fix time
        mesh_out->set_time_step(req_id[0]);
        mesh_out->set_time(this->internals->indices[req_id[0]].time);
    }

    return mesh_out;
}
