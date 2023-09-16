#include "array_scalar_multiply_internals.h"
#include "array.h"

namespace array_scalar_multiply_internals
{
// **************************************************************************
int cpu_dispatch(p_array &result, const const_p_array &array_in,
    double scalar, size_t n_vals)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_scalar_multiply_internals::cpu_dispatch" << std::endl;
#endif
    // ensure the data is accessible on the CPU
    std::shared_ptr<const double> parray_in = array_in->get_host_accessible();

    // allocate the result
    result = array::new_host_accessible();
    result->resize(n_vals);

    // do the calculation
    array_scalar_multiply_internals::cpu::multiply(
        result->data(), parray_in.get(), scalar, n_vals);

    return 0;
}

#if !defined(TECA_HAS_CUDA)
// **************************************************************************
int cuda_dispatch(int device_id, p_array &result,
    const const_p_array &array_in, double scalar, size_t n_vals)
{
    (void) device_id;
    (void) result;
    (void) array_in;
    (void) scalar;
    (void) n_vals;

    TECA_ERROR("array_scalar_multiply failed because CUDA is not available")

    return -1;
}
#endif
}
