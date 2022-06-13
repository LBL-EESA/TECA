#ifndef array_source_internals_h
#define array_source_internals_h

#include "array.h"

namespace array_source_internals
{
namespace cpu
{
template<typename data_t>
void initialize(data_t *data, double val, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
        data[i] = val;
}
}

/// execute on the CPU
int cpu_dispatch(p_array &a_out, double val, size_t n_vals);

/// execute on the GPU
int cuda_dispatch(int device_id, p_array &a_out, double val, size_t n_vals);

}
#endif
