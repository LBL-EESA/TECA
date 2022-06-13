#ifndef array_add_internals_h
#define array_add_internals_h

#include "array.h"

namespace array_add_internals
{
namespace cpu
{
// **************************************************************************
template<typename data_t>
void add(data_t *result, const data_t *array_1,
    const data_t *array_2, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
        result[i] = array_1[i] + array_2[i];;
}
}

/// execute on the CPU
int cpu_dispatch(p_array &result, const const_p_array &array_1,
    const const_p_array &array_2, size_t n_vals);

/// execute on the GPU
int cuda_dispatch(int device_id, p_array &result,
    const const_p_array &array_1, const const_p_array &array_2,
    size_t n_vals);

}
#endif
