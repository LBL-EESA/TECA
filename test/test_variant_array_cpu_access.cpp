#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <iostream>
#include <tuple>
#include <memory>

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;


// **************************************************************************
template<typename NT>
void initialize_cpu(NT *data, double val, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        data[i] = val;
    }
}

// **************************************************************************
template <typename NT>
p_teca_variant_array_impl<NT> initialize_cpu(size_t n_vals, const NT &val)
{
    using TT = teca_variant_array_impl<NT>;

    // allocate the memory
    auto [ao, pao] = ::New<TT>(n_vals, allocator::malloc);

    // initialize the data
    initialize_cpu(pao, val, n_vals);

    std::cerr << "initialized to an array of " << n_vals
        << " to " << val << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
        ao->debug_print();
        std::cerr << std::endl;
    }

    return ao;
}



// **************************************************************************
template<typename NT1, typename NT2>
void add_cpu(NT1 *result, const NT1 *array_1, const NT2 *array_2, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_1[i] + array_2[i];
    }
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> add_cpu(
    const const_p_teca_variant_array_impl<NT1> &a1,
    const const_p_teca_variant_array_impl<NT2> &a2)
{
    using TT1 = teca_variant_array_impl<NT1>;
    using TT2 = teca_variant_array_impl<NT2>;

    // get the inputs
    auto [spa1, pa1] = get_host_accessible<const TT1>(a1);
    auto [spa2, pa2] = get_host_accessible<const TT2>(a2);

    // allocate the memory
    size_t n_vals = a1->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::malloc);

    sync_host_access_any(a1, a2);

    // initialize the data
    add_cpu(pao, pa1, pa2, n_vals);

    std::cerr << "added " << n_vals << " array "
        << typeid(NT1).name() << sizeof(NT1) << " to array  "
        << typeid(NT2).name() << sizeof(NT2) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1->debug_print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    return ao;
}



// **************************************************************************
template<typename NT1, typename NT2>
void multiply_scalar_cpu(NT1 *result,
    const NT1 *array_in, NT2 scalar, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_in[i] * scalar;
    }
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> multiply_scalar_cpu(
    const const_p_teca_variant_array_impl<NT1> &ain, const NT2 &val)
{
    using TT1 = teca_variant_array_impl<NT1>;

    // get the inputs
    auto [spain, pain] = get_host_accessible<TT1>(ain);

    // allocate the memory
    size_t n_vals = ain->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::malloc);

    // initialize the data
    multiply_scalar_cpu(pao, pain, val, n_vals);

    std::cerr << "multiply_scalar " << val << " "
         << typeid(NT2).name() << sizeof(NT2) << " by " << n_vals << " array "
         << typeid(NT1).name() << sizeof(NT1) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ain->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    return ao;
}



// **************************************************************************
template <typename T>
int compare_int(const const_p_teca_variant_array_impl<T> &ain, int val)
{
    using TT = teca_variant_array_impl<T>;

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
    teca_variant_array::allocator cpu_alloc = teca_variant_array::allocator::malloc;

    p_teca_float_array  ao0 = teca_float_array::New(n_vals, 1.0f, cpu_alloc);   // = 1 (CPU)
    p_teca_float_array  ao1 = multiply_scalar_cpu(const_ptr(ao0), 2.0f);        // = 2 (CPU)
    ao0 = nullptr;

    p_teca_double_array ao2 = initialize_cpu(n_vals, 2.0);                      // = 2 (CPU)
    p_teca_double_array ao3 = add_cpu(const_ptr(ao2), const_ptr(ao1));          // = 4 (CPU)
    ao1 = nullptr;
    ao2 = nullptr;

    p_teca_double_array ao4 = multiply_scalar_cpu(const_ptr(ao3), 1000.0);      // = 4000 (CPU)
    ao3 = nullptr;

    p_teca_float_array  ao5 = teca_float_array::New(n_vals, 3.0f, cpu_alloc);   // = 1 (CPU)
    p_teca_float_array  ao6 = multiply_scalar_cpu(const_ptr(ao5), 100.0f);      // = 300 (CPU)
    ao5 = nullptr;

    p_teca_float_array ao7 = teca_float_array::New(n_vals, cpu_alloc);          // = uninit (CPU)
    ao7->set(const_ptr(ao6));                                                   // = 300 (CPU)
    ao6 = nullptr;

    p_teca_double_array ao8 = add_cpu(const_ptr(ao4), const_ptr(ao7));          // = 4300 (CPU)
    ao4 = nullptr;
    ao7 = nullptr;

    return compare_int(const_ptr(ao8), 4300);
}

