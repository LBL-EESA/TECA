#ifndef array_scalar_multiply_internals_h
#define array_scalar_multiply_internals_h

#include "array.h"

namespace array_scalar_multiply_internals
{
namespace cpu
{
// **************************************************************************
template<typename data_t>
void multiply(data_t *result, const data_t *array_in,
    data_t scalar, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
        result[i] = array_in[i] * scalar;
}
}

/// execute on the CPU
int cpu_dispatch(p_array &result, const const_p_array &array_in,
    double scalar, size_t n_vals);

/// execute on the GPU
int cuda_dispatch(int device_id, p_array &result,
    const const_p_array &array_in, double scalar, size_t n_vals);
}
#endif
