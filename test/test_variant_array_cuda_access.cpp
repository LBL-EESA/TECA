#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_cuda_util.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

// **************************************************************************
template<typename NT>
__global__
void initialize_cuda(NT *data, double val, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    data[i] = val;
}

// **************************************************************************
template <typename NT, typename TT = teca_variant_array_impl<NT>>
std::shared_ptr<TT> initialize_cuda(size_t n_vals, const NT &val)
{
    // allocate the memory
    auto [ao, pao] = ::New<TT>(n_vals, allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    initialize_cuda<<<block_grid, thread_grid>>>(pao, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "initialized to an array of " << n_vals << " elements of type "
        << typeid(NT).name() << sizeof(NT) << " to " << val << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
        ao->debug_print();
        std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}






// **************************************************************************
template<typename NT1, typename NT2>
__global__
void add_cuda(NT1 *result, const NT1 *array_1, const NT2 *array_2, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_1[i] + array_2[i];
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> add_cuda(const const_p_teca_variant_array_impl<NT1> &a1,
    const const_p_teca_variant_array_impl<NT2> &a2)
{
    using TT1 = teca_variant_array_impl<NT1>;
    using TT2 = teca_variant_array_impl<NT2>;

    // get the inputs
    auto [spa1, pa1] = get_cuda_accessible<TT1>(a1);
    auto [spa2, pa2] = get_cuda_accessible<TT2>(a2);

    // allocate the memory
    size_t n_vals = a1->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    add_cuda<<<block_grid, thread_grid>>>(pao, pa1, pa2, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "added array of " << n_vals << " elements of type "
        << typeid(NT1).name() << sizeof(NT1) << " to array of type "
        << typeid(NT2).name() << sizeof(NT2) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1->debug_print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}





// **************************************************************************
template<typename NT1, typename NT2>
__global__
void multiply_scalar_cuda(NT1 *result, const NT1 *array_in, NT2 scalar, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_in[i] * scalar;
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> multiply_scalar_cuda(
    const const_p_teca_variant_array_impl<NT1> &ain, const NT2 &val)
{
    using TT1 = teca_variant_array_impl<NT1>;

    // get the inputs
    auto [spain, pain] = get_cuda_accessible<TT1>(ain);

    // allocate the memory
    size_t n_vals = ain->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    multiply_scalar_cuda<<<block_grid, thread_grid>>>(pao, pain, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "multiply_scalar " << val << " type "
        << typeid(NT2).name() << sizeof(NT2) << " by array type "
        << typeid(NT1).name() << sizeof(NT1) << " with " << n_vals
        << " elements" << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ain->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}



// **************************************************************************
template <typename NT>
int compare_int(const const_p_teca_variant_array_impl<NT> &ain, int val)
{
    size_t n_vals = ain->size();

    std::cerr << "comparing array with " << n_vals
        << " elements to " << val << std::endl;

    p_teca_int_array ai = teca_int_array::New(n_vals, ain->get_allocator());
    ain->get(ai);

    if (n_vals < 33)
    {
        ai->debug_print();
    }

    auto [spai, pai] = get_host_accessible<teca_int_array>(ai);

    sync_host_access_any(ai);

    for (size_t i = 0; i < n_vals; ++i)
    {
        if (pai[i] != val)
        {
            std::cerr << "ERROR: pai[" << i << "] = " << pai[i]
                << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "all elements are equal to " << val << std::endl;

    return 0;
}




int main(int, char **)
{
    size_t n_vals = 100000;

    allocator cuda_alloc = allocator::cuda_async;
    allocator cpu_alloc = allocator::malloc;

    p_teca_float_array  ao0 = teca_float_array::New(n_vals, 1.0f, cuda_alloc);  // = 1 (CUDA)
    p_teca_float_array  ao1 = multiply_scalar_cuda(const_ptr(ao0), 2.0f);       // = 2 (CUDA)
    ao0 = nullptr;

    p_teca_double_array ao2 = initialize_cuda(n_vals, 2.0);                     // = 2 (CUDA)
    p_teca_double_array ao3 = add_cuda(const_ptr(ao2), const_ptr(ao1));         // = 4 (CUDA)
    ao1 = nullptr;
    ao2 = nullptr;

    p_teca_double_array ao4 = multiply_scalar_cuda(const_ptr(ao3), 1000.0);     // = 4000 (CUDA)
    ao3 = nullptr;

    p_teca_float_array  ao5 = teca_float_array::New(n_vals, 3.0f, cpu_alloc);   // = 1 (CPU)
    p_teca_float_array  ao6 = multiply_scalar_cuda(const_ptr(ao5), 100.0f);     // = 300 (CUDA)
    ao5 = nullptr;

    p_teca_float_array ao7 = teca_float_array::New(n_vals, cpu_alloc);          // = uninit (CPU)
    ao7->set(const_ptr(ao6));                                                   // = 300 (CPU)
    ao6 = nullptr;

    p_teca_double_array ao8 = add_cuda(const_ptr(ao4), const_ptr(ao7));         // = 4300 (CUDA)
    ao4 = nullptr;
    ao7 = nullptr;

    return compare_int(const_ptr(ao8), 4300);
}
