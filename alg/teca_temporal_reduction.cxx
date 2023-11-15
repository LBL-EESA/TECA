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
#include <typeinfo>

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
    const_p_teca_variant_array
    ensure_floating_point(allocator alloc, const const_p_teca_variant_array &array);

public:
    class reduction_operator;
    class average_operator;
    class summation_operator;
    class minimum_operator;
    class maximum_operator;
    class reduction_operator_factory;
    class time_interval;

    using p_reduction_operator = std::shared_ptr<reduction_operator>;

    void set_operation(const std::string &array, const p_reduction_operator &op);
    p_reduction_operator &get_operation(const std::string &array);


public:
    std::mutex m_mutex;
    teca_metadata metadata;
    std::vector<time_interval> indices;
    std::map<std::thread::id, std::map<std::string, p_reduction_operator>> operation;
};

// --------------------------------------------------------------------------
void teca_cpp_temporal_reduction::internals_t::set_operation(
    const std::string &array, const p_reduction_operator &op)
{
    auto tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(m_mutex);
    this->operation[tid][array] = op;
}

// --------------------------------------------------------------------------
teca_cpp_temporal_reduction::internals_t::p_reduction_operator &
teca_cpp_temporal_reduction::internals_t::get_operation(const std::string &array)
{
    auto tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(m_mutex);
    return this->operation[tid][array];
}

// --------------------------------------------------------------------------
const_p_teca_variant_array
teca_cpp_temporal_reduction::internals_t::ensure_floating_point(
     allocator alloc, const const_p_teca_variant_array &array)
{
    if (std::dynamic_pointer_cast<const teca_variant_array_impl<double>>(array)  ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<float>>(array))
    {
        // the data is already floating point type
        return array;
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
        return tmp;
    }
    else
    {
        // convert from a 32 bit integer to a 32 bit floating point
        size_t n_elem = array->size();
        p_teca_float_array tmp = teca_float_array::New(n_elem, alloc);
        tmp->set(0, array, 0, n_elem);
        return tmp;
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
    virtual ~reduction_operator() {}

    reduction_operator() : fill_value(-1), result(nullptr),
                           valid(nullptr), count(nullptr) {}

    virtual void initialize(double fill_value);

    virtual int init(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) = 0;

    virtual int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) = 0;

    virtual int update_gpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) = 0;

    virtual int finalize(int device_id);

public:
    double fill_value;
    p_teca_variant_array result;
    p_teca_variant_array valid;
    p_teca_variant_array count;
};


/// implements time average
class teca_cpp_temporal_reduction::internals_t::average_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int init(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_gpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int finalize(int device_id) override;
};


/// implements sum over time
class teca_cpp_temporal_reduction::internals_t::summation_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int init(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_gpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;
};


/// implements minimum over time
class teca_cpp_temporal_reduction::internals_t::minimum_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int init(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_gpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;
};


/// implements maximum over time
class teca_cpp_temporal_reduction::internals_t::maximum_operator :
      public teca_cpp_temporal_reduction::internals_t::reduction_operator
{
public:
    int init(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;

    int update_gpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long steps_per_request) override;
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
class teca_cpp_temporal_reduction::internals_t::time_interval
{
public:
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
namespace cuda_gpu
{
// --------------------------------------------------------------------------
template <typename T>
__global__
void average_initialize(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     T *p_res_count,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_in_valid == nullptr)
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] += p_in_array[i+j*n_elem];
        }
        p_res_count[0] = steps_per_request;
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem])
            {
                p_res_array[i] += p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
                p_res_count[i] += 1;
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int average_initialize(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    T *p_res_count,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    average_initialize<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                                 p_res_array, p_res_valid, p_res_count,
                                                 n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the average_initialize CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T, typename T_RES>
__global__
void average(
     const T *p_in_array, const char *p_in_valid,
     T_RES *p_res_count, T_RES *p_res_array, char *p_res_valid,
     unsigned long n_elem, unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_res_valid == nullptr)
    {
        // accumulate
        if (i == 0) p_res_count[0] += steps_per_request;
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] += p_in_array[i+j*n_elem];
        }
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            //update the count only where there is valid data
            if (p_in_valid[i+j*n_elem])
            {
                p_res_array[i] += p_in_array[i+j*n_elem];
                p_res_count[i] += 1;
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T, typename T_RES>
int average(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T_RES *p_res_count, T_RES *p_res_array, char *p_res_valid,
    unsigned long n_elem, unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    average<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                        p_res_count, p_res_array, p_res_valid,
                                        n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the average CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void summation_initialize(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_in_valid == nullptr)
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] += p_in_array[i+j*n_elem];
        }
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem])
            {
                p_res_array[i] += p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int summation_initialize(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    summation_initialize<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                                   p_res_array, p_res_valid,
                                                   n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the summation_initialize CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T, typename T_RES>
__global__
void summation(
     const T *p_in_array, const char *p_in_valid,
     T_RES *p_res_array, char *p_res_valid,
     unsigned long n_elem, unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_res_valid == nullptr)
    {
        // accumulate
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] += p_in_array[i+j*n_elem];
        }
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem])
            {
                p_res_array[i] += p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T, typename T_RES>
int summation(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T_RES *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    summation<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                          p_res_array, p_res_valid,
                                          n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the summation CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void minimum_initialize(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_in_valid == nullptr)
    {
        p_res_array[i] = p_in_array[i];
        for (unsigned int j = 1; j < steps_per_request; ++j)
        {
            p_res_array[i] =
                  (p_in_array[i+j*n_elem] < p_res_array[i]) ?
                   p_in_array[i+j*n_elem] : p_res_array[i];
        }
    }
    else
    {
        p_res_array[i] = p_in_array[i];
        p_res_valid[i] = (!p_in_valid[i]) ? char(0) : char(1);
        for (unsigned int j = 1; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem] && p_res_valid[i])
            {
                p_res_array[i] =
                      (p_in_array[i+j*n_elem] < p_res_array[i]) ?
                       p_in_array[i+j*n_elem] : p_res_array[i];
            }
            else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
            {
                p_res_array[i] = p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int minimum_initialize(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    minimum_initialize<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                             p_res_array, p_res_valid,
                                             n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the minimum_initialize CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void minimum(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_res_valid == nullptr)
    {
        // reduce
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] = (p_in_array[i+j*n_elem] < p_res_array[i]) ?
                              p_in_array[i+j*n_elem] : p_res_array[i];
        }
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem] && p_res_valid[i])
            {
                p_res_array[i] = (p_in_array[i+j*n_elem] < p_res_array[i]) ?
                                  p_in_array[i+j*n_elem] : p_res_array[i];
            }
            else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
            {
                p_res_array[i] = p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int minimum(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    minimum<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                        p_res_array, p_res_valid,
                                        n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the minimum CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void maximum_initialize(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_in_valid == nullptr)
    {
        p_res_array[i] = p_in_array[i];
        for (unsigned int j = 1; j < steps_per_request; ++j)
        {
            p_res_array[i] =
                  (p_in_array[i+j*n_elem] > p_res_array[i]) ?
                   p_in_array[i+j*n_elem] : p_res_array[i];
        }
    }
    else
    {
        p_res_array[i] = p_in_array[i];
        p_res_valid[i] = (!p_in_valid[i]) ? char(0) : char(1);
        for (unsigned int j = 1; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem] && p_res_valid[i])
            {
                p_res_array[i] =
                      (p_in_array[i+j*n_elem] > p_res_array[i]) ?
                       p_in_array[i+j*n_elem] : p_res_array[i];
            }
            else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
            {
                p_res_array[i] = p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int maximum_initialize(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    maximum_initialize<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                                 p_res_array, p_res_valid,
                                                 n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the maximum_initialize CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void maximum(
     const T *p_in_array, const char *p_in_valid,
     T *p_res_array, char *p_res_valid,
     unsigned long n_elem,
     unsigned long steps_per_request)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_res_valid == nullptr)
    {
        // reduce
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            p_res_array[i] = (p_in_array[i+j*n_elem] > p_res_array[i]) ?
                              p_in_array[i+j*n_elem] : p_res_array[i];
        }
    }
    else
    {
        for (unsigned int j = 0; j < steps_per_request; ++j)
        {
            if (p_in_valid[i+j*n_elem] && p_res_valid[i])
            {
                p_res_array[i] = (p_in_array[i+j*n_elem] > p_res_array[i]) ?
                                  p_in_array[i+j*n_elem] : p_res_array[i];
            }
            else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
            {
                p_res_array[i] = p_in_array[i+j*n_elem];
                p_res_valid[i] = char(1);
            }
        }
    }
}

// --------------------------------------------------------------------------
template <typename T>
int maximum(
    int device_id,
    const T *p_in_array, const char *p_in_valid,
    T *p_res_array, char *p_res_valid,
    unsigned long n_elem,
    unsigned long steps_per_request)
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

    cudaError_t ierr = cudaSuccess;
    maximum<<<block_grid,thread_grid>>>(p_in_array, p_in_valid,
                                        p_res_array, p_res_valid,
                                        n_elem, steps_per_request);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the maximum CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
__global__
void finalize(
     T *p_res_array, const char *p_res_valid,
     double fill_value,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (!p_res_valid[i])
    {
        p_res_array[i] = fill_value;
    }
}

// --------------------------------------------------------------------------
template <typename T>
int finalize(
    int device_id,
    T *p_res_array, const char *p_res_valid,
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

    cudaError_t ierr = cudaSuccess;
    finalize<<<block_grid,thread_grid>>>(p_res_array, p_res_valid,
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
     T *p_res_array, const char *p_res_valid,
     const T *p_res_count,
     double fill_value,
     unsigned long n_elem)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    if (p_res_valid == nullptr)
    {
        p_res_array[i] = p_res_array[i]/p_res_count[0];
    }
    else
    {
        p_res_array[i] = (!p_res_valid[i]) ?
                                fill_value : p_res_array[i]/p_res_count[i];
    }
}

// --------------------------------------------------------------------------
template <typename T>
int average_finalize(
    int device_id,
    T *p_res_array, const char *p_res_valid,
    const T *p_res_count,
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

    cudaError_t ierr = cudaSuccess;
    average_finalize<<<block_grid,thread_grid>>>(p_res_array, p_res_valid,
                                                 p_res_count, fill_value, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the average_finalize CUDA kernel: "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}
} //end of namespace cuda
#endif

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::init(
    int device_id,
    const const_p_teca_variant_array &input_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    allocator alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
        alloc = allocator::cuda_async;
#endif

    // don't use integer types for this calculation
    auto in_array = internals_t::ensure_floating_point(alloc, input_array);

    unsigned long n_elem = in_array->size();
    unsigned long n_elem_per_timestep = n_elem/steps_per_request;

    // get the current count
    size_t n_elem_count = in_valid ? n_elem_per_timestep : 1;

    VARIANT_ARRAY_DISPATCH(in_array.get(),

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            // GPU
            auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep, 0.,
                                                      allocator::cuda_async);
            this->count = teca_long_array::New(n_elem_count, NT(0),
                                                      allocator::cuda_async);

            auto [p_res_array] = data<TT>(this->result);
            auto [p_res_count] = data<TT>(this->count);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep, NT_MASK(0),
                                                               allocator::cuda_async);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::average_initialize(device_id, p_in_array, p_in_valid,
                                           p_res_array, p_res_valid, p_res_count,
                                           n_elem_per_timestep, steps_per_request);
            }
            else
            {
                cuda_gpu::average_initialize(device_id, p_in_array, nullptr,
                                           p_res_array, nullptr, p_res_count,
                                           n_elem_per_timestep, steps_per_request);
            }
        }
        else
        {
#endif
            // CPU
            auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep, 0.);
            this->count = teca_long_array::New(n_elem_count, NT(0));

            auto [p_res_array] = data<TT>(this->result);
            auto [p_res_count] = data<TT>(this->count);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep, NT_MASK(0));
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                sync_host_access_any(in_array, in_valid);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        if (p_in_valid[i+j*n_elem_per_timestep])
                        {
                            p_res_array[i] += p_in_array[i+j*n_elem_per_timestep];
                            p_res_valid[i] = NT_MASK(1);
                            p_res_count[i] += NT(1);
                        }
                    }
                }
            }
            else
            {
                sync_host_access_any(in_array);

                p_res_count[0] = NT(steps_per_request);
                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        p_res_array[i] += p_in_array[i+j*n_elem_per_timestep];
                    }
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::update_cpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
    (void)device_id;

    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),
        NESTED_VARIANT_ARRAY_DISPATCH_FP(this->result.get(), _RES,

            auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);

            auto [p_res_array] = data<TT_RES>(this->result);
            auto [p_res_count] = data<TT_RES>(this->count);

            if (this->valid)
            {
                // update, respecting missing data
                auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                sync_host_access_any(in_array, in_valid);

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        // update, respecting missing data
                        // count where there is valid data,
                        // otherwise pass the current count through
                        if (p_in_valid[i+j*n_elem])
                        {
                            p_res_array[i] += p_in_array[i+j*n_elem];
                            p_res_count[i] += NT_RES(1);
                            p_res_valid[i] = NT_MASK(1);
                        }
                    }
                }
            }
            else
            {
                sync_host_access_any(in_array);

                // update, no missing data
                p_res_count[0] += NT_RES(steps_per_request);
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        p_res_array[i] += p_in_array[i+j*n_elem];
                    }
                }
            }
        )
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::update_gpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if defined(TECA_HAS_CUDA)
    unsigned long n_elem = this->result->size();

    // allocate space for the output mask which is a function of the current
    // input and previous state and size and place the count array
    size_t n_elem_count = in_valid ? n_elem : 1;

    VARIANT_ARRAY_DISPATCH(in_array.get(),
        NESTED_VARIANT_ARRAY_DISPATCH_FP(this->result.get(), _RES,

            auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
            auto [p_res_array] = data<TT_RES>(this->result);
            auto [p_res_count] = data<TT_RES>(this->count);

            if (this->valid)
            {
                // update while respecting invalid/missing data
                auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::average(device_id, p_in_array, p_in_valid, p_res_count,
                    p_res_array, p_res_valid, n_elem, steps_per_request);
            }
            else
            {
                // update, no missing data
                cuda_gpu::average(device_id, p_in_array, nullptr, p_res_count,
                    p_res_array, nullptr, n_elem, steps_per_request);
            }
        )
    )

    return 0;
#else
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::summation_operator::init(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    // the valid value masks
    unsigned long n_elem = in_array->size();
    unsigned long n_elem_per_timestep = n_elem/steps_per_request;

    VARIANT_ARRAY_DISPATCH(in_array.get(),

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            // GPU
            auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep, 0.,
                                                      allocator::cuda_async);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep, NT_MASK(0),
                                                               allocator::cuda_async);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::summation_initialize(device_id, p_in_array, p_in_valid,
                                           p_res_array, p_res_valid,
                                           n_elem_per_timestep, steps_per_request);
            }
            else
            {
                cuda_gpu::summation_initialize(device_id, p_in_array, nullptr,
                                           p_res_array, nullptr,
                                           n_elem_per_timestep, steps_per_request);
            }
        }
        else
        {
#endif
            // CPU
            auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep, 0.);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep, NT_MASK(0));
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                sync_host_access_any(in_array, in_valid);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        if (p_in_valid[i+j*n_elem_per_timestep])
                        {
                            p_res_array[i] += p_in_array[i+j*n_elem_per_timestep];
                            p_res_valid[i] = NT_MASK(1);
                        }
                    }
                }
            }
            else
            {
                sync_host_access_any(in_array);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    for (unsigned int j = 0; j < steps_per_request; ++j)
                    {
                        p_res_array[i] += p_in_array[i+j*n_elem_per_timestep];
                    }
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::summation_operator::update_cpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
    (void)device_id;

    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update, respecting missing data
            auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            sync_host_access_any(in_array, in_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    // update, respecting missing data
                    if (p_in_valid[i+j*n_elem])
                    {
                        p_res_array[i] += p_in_array[i+j*n_elem];
                        p_res_valid[i] = NT_MASK(1);
                    }
                }
            }
        }
        else
        {
            sync_host_access_any(in_array);

            // update, no missing data
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    p_res_array[i] += p_in_array[i+j*n_elem];
                }
            }
        }
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::summation_operator::update_gpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if defined(TECA_HAS_CUDA)
    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update while respecting invalid/missing data
            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            cuda_gpu::summation(device_id, p_in_array, p_in_valid,
                p_res_array, p_res_valid, n_elem, steps_per_request);
        }
        else
        {
            cuda_gpu::summation(device_id, p_in_array, nullptr,
                p_res_array, nullptr, n_elem, steps_per_request);
        }
    )

    return 0;
#else
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::minimum_operator::init(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
#endif
    unsigned long n_elem = in_array->size();
    unsigned long n_elem_per_timestep = n_elem/steps_per_request;

    VARIANT_ARRAY_DISPATCH(in_array.get(),

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep,
                                                      allocator::cuda_async);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep,
                                                               allocator::cuda_async);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::minimum_initialize(device_id, p_in_array, p_in_valid,
                                           p_res_array, p_res_valid,
                                           n_elem_per_timestep, steps_per_request);
            }
            else
            {
                cuda_gpu::minimum_initialize(device_id, p_in_array, nullptr,
                                           p_res_array, nullptr,
                                           n_elem_per_timestep, steps_per_request);
            }
        }
        else
        {
#endif
            // CPU
            auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                sync_host_access_any(in_array, in_valid);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    p_res_array[i] = p_in_array[i];
                    p_res_valid[i] = (!p_in_valid[i]) ? NT_MASK(0) : NT_MASK(1);
                    for (unsigned int j = 1; j < steps_per_request; ++j)
                    {
                        if (p_in_valid[i+j*n_elem_per_timestep] &&
                            p_res_valid[i])
                        {
                            p_res_array[i] =
                              (p_in_array[i+j*n_elem_per_timestep] < p_res_array[i]) ?
                               p_in_array[i+j*n_elem_per_timestep] : p_res_array[i];
                        }
                        else if (p_in_valid[i+j*n_elem_per_timestep] &&
                                !p_res_valid[i])
                        {
                            p_res_array[i] = p_in_array[i+j*n_elem_per_timestep];
                            p_res_valid[i] = NT_MASK(1);
                        }
                    }
                }
            }
            else
            {
                sync_host_access_any(in_array);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    p_res_array[i] = p_in_array[i];
                    for (unsigned int j = 1; j < steps_per_request; ++j)
                    {
                        p_res_array[i] =
                          (p_in_array[i+j*n_elem_per_timestep] < p_res_array[i]) ?
                           p_in_array[i+j*n_elem_per_timestep] : p_res_array[i];
                    }
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::minimum_operator::update_cpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
    (void)device_id;

    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update, respecting missing data
            auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            sync_host_access_any(in_array, in_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    if (p_in_valid[i+j*n_elem] && p_res_valid[i])
                    {
                        p_res_array[i] = (p_in_array[i+j*n_elem] < p_res_array[i]) ?
                                          p_in_array[i+j*n_elem] : p_res_array[i];
                    }
                    else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
                    {
                        p_res_array[i] = p_in_array[i+j*n_elem];
                        p_res_valid[i] = NT_MASK(1);
                    }
                }
            }
        }
        else
        {
            sync_host_access_any(in_array);

            // update, no missing values
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    p_res_array[i] = p_in_array[i+j*n_elem] < p_res_array[i] ?
                                     p_in_array[i+j*n_elem] : p_res_array[i];
                }
            }
        }
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::minimum_operator::update_gpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if defined(TECA_HAS_CUDA)
    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update while respecting invalid/missing data
            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            cuda_gpu::minimum(device_id, p_in_array, p_in_valid,
                p_res_array, p_res_valid, n_elem, steps_per_request);

        }
        else
        {
            cuda_gpu::minimum(device_id, p_in_array, nullptr,
                p_res_array, nullptr, n_elem, steps_per_request);
        }
    )

    return 0;
#else
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::maximum_operator::init(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
#endif
    unsigned long n_elem = in_array->size();
    unsigned long n_elem_per_timestep = n_elem/steps_per_request;

    VARIANT_ARRAY_DISPATCH(in_array.get(),

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep,
                                                      allocator::cuda_async);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep,
                                                               allocator::cuda_async);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::maximum_initialize(device_id, p_in_array, p_in_valid,
                                           p_res_array, p_res_valid,
                                           n_elem_per_timestep, steps_per_request);
            }
            else
            {
                cuda_gpu::maximum_initialize(device_id, p_in_array, nullptr,
                                           p_res_array, nullptr,
                                           n_elem_per_timestep, steps_per_request);
            }
        }
        else
        {
#endif
            // CPU
            auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);

            this->result = teca_variant_array_impl<NT>::New(n_elem_per_timestep);
            auto [p_res_array] = data<TT>(this->result);

            if (in_valid)
            {
                auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);

                this->valid = teca_variant_array_impl<NT_MASK>::New(n_elem_per_timestep);
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                sync_host_access_any(in_array, in_valid);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    p_res_array[i] = p_in_array[i];
                    p_res_valid[i] = (!p_in_valid[i]) ? NT_MASK(0) : NT_MASK(1);
                    for (unsigned int j = 1; j < steps_per_request; ++j)
                    {
                        if (p_in_valid[i+j*n_elem_per_timestep] &&
                            p_res_valid[i])
                        {
                            p_res_array[i] =
                              (p_in_array[i+j*n_elem_per_timestep] > p_res_array[i]) ?
                               p_in_array[i+j*n_elem_per_timestep] : p_res_array[i];
                        }
                        else if (p_in_valid[i+j*n_elem_per_timestep] &&
                                !p_res_valid[i])
                        {
                            p_res_array[i] = p_in_array[i+j*n_elem_per_timestep];
                            p_res_valid[i] = NT_MASK(1);
                        }
                    }
                }
            }
            else
            {
                sync_host_access_any(in_array);

                for (unsigned int i = 0; i < n_elem_per_timestep; ++i)
                {
                    p_res_array[i] = p_in_array[i];
                    for (unsigned int j = 1; j < steps_per_request; ++j)
                    {
                        p_res_array[i] =
                          (p_in_array[i+j*n_elem_per_timestep] > p_res_array[i]) ?
                           p_in_array[i+j*n_elem_per_timestep] : p_res_array[i];
                    }
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::maximum_operator::update_cpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
    (void)device_id;

    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update, respecting missing data
            auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            sync_host_access_any(in_array, in_valid);

            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    if (p_in_valid[i+j*n_elem] && p_res_valid[i])
                    {
                        p_res_array[i] = (p_in_array[i+j*n_elem] > p_res_array[i]) ?
                                          p_in_array[i+j*n_elem] : p_res_array[i];
                    }
                    else if (p_in_valid[i+j*n_elem] && !p_res_valid[i])
                    {
                        p_res_array[i] = p_in_array[i+j*n_elem];
                        p_res_valid[i] = NT_MASK(1);
                    }
                }
            }
        }
        else
        {
            sync_host_access_any(in_array);

            // update, no missing values
            for (unsigned int i = 0; i < n_elem; ++i)
            {
                for (unsigned int j = 0; j < steps_per_request; ++j)
                {
                    p_res_array[i] = p_in_array[i+j*n_elem] > p_res_array[i] ?
                                     p_in_array[i+j*n_elem] : p_res_array[i];
                }
            }
        }
    )

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::maximum_operator::update_gpu(
    int device_id,
    const const_p_teca_variant_array &in_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long steps_per_request)
{
#if defined(TECA_HAS_CUDA)
    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(in_array.get(),

        auto [sp_in_array, p_in_array] = get_cuda_accessible<CTT>(in_array);
        auto [p_res_array] = data<TT>(this->result);

        if (this->valid)
        {
            // update while respecting invalid/missing data
            auto [sp_in_valid, p_in_valid] = get_cuda_accessible<CTT_MASK>(in_valid);
            auto [p_res_valid] = data<TT_MASK>(this->valid);

            cuda_gpu::maximum(device_id, p_in_array, p_in_valid,
                p_res_array, p_res_valid, n_elem, steps_per_request);

        }
        else
        {
            cuda_gpu::maximum(device_id, p_in_array, nullptr,
                p_res_array, nullptr, n_elem, steps_per_request);
        }
    )

    return 0;
#else
    (void)device_id;
    (void)in_array;
    (void)in_valid;
    (void)steps_per_request;
    TECA_ERROR("CUDA support is not available")
    return -1;
#endif
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::reduction_operator::finalize(
    int device_id)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    unsigned long n_elem = this->result->size();

    if (this->valid)
    {
        VARIANT_ARRAY_DISPATCH(this->result.get(),

            NT fill_value = NT(this->fill_value);

            auto [p_res_array] = data<TT>(this->result);

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::finalize(device_id, p_res_array, p_res_valid,
                                   fill_value, n_elem);
            }
            else
            {
#endif
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_res_array[i] = p_res_valid[i] ? p_res_array[i] : fill_value;
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
        )
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cpp_temporal_reduction::internals_t::average_operator::finalize(
    int device_id)
{
#if !defined(TECA_HAS_CUDA)
    (void)device_id;
#endif

    unsigned long n_elem = this->result->size();

    VARIANT_ARRAY_DISPATCH(this->result.get(),

        NT fill_value = NT(this->fill_value);

        auto [p_res_array] = data<TT>(this->result);
        auto [p_res_count] = data<TT>(this->count);

#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            // GPU
            if (this->valid)
            {
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                cuda_gpu::average_finalize(device_id, p_res_array, p_res_valid,
                                           p_res_count, fill_value, n_elem);
            }
            else
            {
                cuda_gpu::average_finalize(device_id, p_res_array, nullptr,
                                           p_res_count, fill_value, n_elem);
            }
        }
        else
        {
#endif
            // CPU
            if (this->valid)
            {
                auto [p_res_valid] = data<TT_MASK>(this->valid);

                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_res_array[i] = (!p_res_valid[i]) ?
                                      fill_value : p_res_array[i]/p_res_count[i];
                }
            }
            else
            {
                for (unsigned int i = 0; i < n_elem; ++i)
                {
                    p_res_array[i] = p_res_array[i]/p_res_count[0];
                }
            }
#if defined(TECA_HAS_CUDA)
        }
#endif
    )

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
int teca_cpp_temporal_reduction::set_operation(const std::string &op)
{
    if (op == "average")
    {
        this->operation = average;
    }
    else if (op == "summation")
    {
        this->operation = summation;
    }
    else if (op == "minimum")
    {
        this->operation = minimum;
    }
    else if (op == "maximum")
    {
        this->operation = maximum;
    }
    else
    {
        TECA_FATAL_ERROR("Invalid operator name \"" << op << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::string teca_cpp_temporal_reduction::get_operation_name()
{
    std::string name;
    switch(this->operation)
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
            TECA_FATAL_ERROR("Invalid \"operator\" " << this->operation)
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
        TECA_FATAL_ERROR("Invalid interval name \"" << interval << "\"")
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
            TECA_FATAL_ERROR("Invalid \"interval\" " << this->interval)
    }
    return name;
}

// --------------------------------------------------------------------------
teca_cpp_temporal_reduction::teca_cpp_temporal_reduction() :
    operation(average), interval(monthly), number_of_steps(0), fill_value(-1),
    steps_per_request(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->set_stream_size(1);

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
        TECA_POPTS_GET(int, prefix, operation,
            "reduction operator to use"
            " (summation, minimum, maximum, or average)")
        TECA_POPTS_GET(int, prefix, interval,
            "interval to reduce the time axis to"
            " (daily, monthly, seasonal, yearly, n_steps, all)")
        TECA_POPTS_GET(long, prefix, number_of_steps,
            "desired number of steps for the n_steps interval")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        TECA_POPTS_GET(long, prefix, steps_per_request,
            "")
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
    TECA_POPTS_SET(opts, int, prefix, operation)
    TECA_POPTS_SET(opts, int, prefix, interval)
    TECA_POPTS_SET(opts, long, prefix, number_of_steps)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
    TECA_POPTS_SET(opts, long, prefix, steps_per_request)
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

    // use cached the metadata. the first pass is always a single thread.  if
    // we were to generate the metadata at each pass, the code below will have
    // to be serialized.
    if (!this->internals->metadata.empty())
        return this->internals->metadata;

    // sanity checks
    if (this->point_arrays.empty())
    {
        TECA_FATAL_ERROR("No point arrays were specified.")
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
        TECA_FATAL_ERROR("Failed to get units")
        return teca_metadata();
    }

    teca_calendar_util::p_interval_iterator it;
    it = teca_calendar_util::interval_iterator_factory::New(
                                            this->interval);

    if (!it)
    {
        TECA_FATAL_ERROR("Failed to allocate an instance of the \""
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
            TECA_FATAL_ERROR("Failed to set the number of steps")
            return teca_metadata();
        }
    }

    if (it->initialize(t, t_units, cal, 0, -1))
    {
        TECA_FATAL_ERROR("Failed to initialize the \""
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
            TECA_FATAL_ERROR("Failed to get the attributes for \""
                << array << "\"")
            return teca_metadata();
        }

        // convert integer to floating point for averaging operations
        if (this->operation == average)
        {
            int tc;
            if (in_atts.get("type_code", tc))
            {
                TECA_FATAL_ERROR("Failed to get type_code")
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
            << this->get_operation_name() << " of " << array;
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

    // package it all up
    md_out.set("variables", out_vars);
    md_out.set("attributes", out_atts);

    // cache the metadata
    this->internals->metadata = md_out;

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
        if (vars_in.count(vv_mask) && !req_arrays.count(vv_mask))
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
        internals_t::p_reduction_operator op
             = internals_t::reduction_operator_factory::New(this->operation);

        op->initialize(fill_value);

        // save the operator
        this->internals->set_operation(array, op);
    }

    // generate one request for each time step in the interval
    std::vector<teca_metadata> up_reqs;

    std::string request_key;
    if (md.get("index_request_key", request_key))
    {
        TECA_FATAL_ERROR("Failed to locate the index_request_key")
        return up_reqs;
    }

    unsigned long req_id[2];
    if (req_in.get(request_key, req_id))
    {
        TECA_FATAL_ERROR("Failed to get the requested index using the"
            " index_request_key \"" << request_key << "\"")
        return up_reqs;
    }

    if (req_id[0] >= this->internals->indices.size())
    {
        TECA_FATAL_ERROR("Request for step " << req_id[0]
            << " is out of bounds in output with "
            << this->internals->indices.size() << " steps")
        return up_reqs;
    }

    unsigned long steps_per_request = this->steps_per_request;
    int i = this->internals->indices[req_id[0]].start_index;
    int end = this->internals->indices[req_id[0]].end_index;
    while (i <= end)
    {
        teca_metadata req(req_in);
        req.set("arrays", req_arrays);
        int j = i + steps_per_request - 1;
        if (j > end) { j = end; }
        req.set(request_key, {i, j});
        up_reqs.push_back(req);
        i += steps_per_request;
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
        TECA_FATAL_ERROR("metadata issue. failed to get the requested indices")
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

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
       if (teca_cuda_util::set_device(device_id))
           return nullptr;
    }
#endif

    long n_data = data_in.size();
    long data_start = 0;

    // initialize the operator on the first pass.
    if (!this->internals->get_operation(this->point_arrays[0])->result)
    {
        data_start = 1;

        auto mesh_in = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[0]);
        auto arrays_in = mesh_in->get_point_arrays();

        unsigned long t_ext[2] = {0ul};
        mesh_in->get_temporal_extent(t_ext);
        unsigned long steps_per_request = t_ext[1] - t_ext[0] + 1;

        size_t n_array = this->point_arrays.size();
        for (size_t i = 0; i < n_array; ++i)
        {
            const std::string &array = this->point_arrays[i];
            auto &op = this->internals->get_operation(array);

            // get the incoming data array
            auto array_in = arrays_in->get(array);
            if (!array_in)
            {
                TECA_FATAL_ERROR("array \"" << array << "\" not found")
                return nullptr;
            }

            // get the incoming valid value mask
            std::string valid = array + "_valid";
            auto valid_in = arrays_in->get(valid);

            // initialize the operator
            op->init(device_id, array_in, valid_in, steps_per_request);
        }
    }

    // accumulate incoming data
    for (long i = data_start; i < n_data; ++i)
    {
        // get the incoming mesh
        auto mesh_in = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[i]);
        auto const& arrays_in = mesh_in->get_point_arrays();

        // get the incoming number of steps
        unsigned long t_ext[2] = {0ul};
        mesh_in->get_temporal_extent(t_ext);
        unsigned long steps_per_request = t_ext[1] - t_ext[0] + 1;

        size_t n_array = this->point_arrays.size();
        for (size_t j = 0; j < n_array; ++j)
        {
            const std::string &array = this->point_arrays[j];

            // get the incoming data array
            auto array_in = arrays_in->get(array);
            if (!array_in)
            {
                TECA_FATAL_ERROR("array \"" << array << "\" not found")
                return nullptr;
            }

            // get the incoming valid value mask
            std::string valid = array + "_valid";
            auto valid_in = arrays_in->get(valid);

            // apply the reduction
            auto &op = this->internals->get_operation(array);

#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                op->update_gpu(device_id, array_in, valid_in, steps_per_request);
            }
            else
            {
#endif
                op->update_cpu(device_id, array_in, valid_in, steps_per_request);
#if defined(TECA_HAS_CUDA)
            }
#endif
        }
    }

    // when all the data is processed finalize the reduction
    p_teca_cartesian_mesh mesh_out;

    if (!streaming)
    {
        auto mesh_in = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[0]);

        mesh_out = teca_cartesian_mesh::New();
        mesh_out->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));

        auto &arrays_out = mesh_out->get_point_arrays();

        size_t n_array = this->point_arrays.size();
        for (size_t i = 0; i < n_array; ++i)
        {
            const std::string &array = this->point_arrays[i];
            auto &op = this->internals->get_operation(array);

            // finalize the calculation
            op->finalize(device_id);

            // pass the result
            arrays_out->set(array, op->result);

            if (op->valid)
                arrays_out->set(array + "_valid", op->valid);
        }

        // fix time
        mesh_out->set_time_step(req_id[0]);
        mesh_out->set_time(this->internals->indices[req_id[0]].time);
    }

    return mesh_out;
}
