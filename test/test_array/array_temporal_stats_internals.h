#ifndef array_temporal_stats_internals_h
#define array_temporal_stats_internals_h

#include "array.h"

namespace array_temporal_stats_internals
{
namespace cpu
{
/// allocate and initialize results on the CPU
p_array allocate_and_initialize_stats(const std::string &array_name);

// **************************************************************************
template<typename data_t>
void compute_stats(data_t *array_out, const data_t *array_in, size_t n_vals)
{
    // min
    data_t min_in = std::numeric_limits<data_t>::max();

    for (size_t i = 0; i < n_vals; ++i)
        min_in = min_in > array_in[i] ? array_in[i] : min_in;

    array_out[0] = min_in;

    // max
    data_t max_in = std::numeric_limits<data_t>::lowest();

    for (size_t i = 0; i < n_vals; ++i)
        max_in = max_in < array_in[i] ? array_in[i] : max_in;

    array_out[1] = max_in;

    // sum
    data_t sum_in = data_t(0);

    for (size_t i = 0; i < n_vals; ++i)
        sum_in += array_in[i];

    array_out[2] = sum_in;

    // count
    array_out[3] = data_t(n_vals);
}

// **************************************************************************
template<typename data_t>
void reduce_stats(data_t *array_out, const data_t *array_in_l, const data_t *array_in_r)
{
    // min
    array_out[0] = array_in_l[0] < array_in_r[0] ? array_in_l[0] : array_in_r[0];

    // max
    array_out[1] = array_in_l[1] > array_in_r[1] ? array_in_l[1] : array_in_r[1];

    // sum
    array_out[2] = array_in_l[2] + array_in_r[2];

    // count
    array_out[3] = array_in_l[3] + array_in_r[3];
}

// **************************************************************************
template<typename data_t>
void initialize_stats(data_t *array_out)
{
    array_out[0] = std::numeric_limits<data_t>::max();
    array_out[1] = std::numeric_limits<data_t>::lowest();
    array_out[2] = data_t(0);
    array_out[3] = data_t(0);
}
}

/// execute on the CPU
int cpu_dispatch(p_array &result, const const_p_array &l_in,
    const const_p_array &r_in, bool l_active, bool r_active);

/// execute on the GPU
int cuda_dispatch(int device_id, p_array &result, const const_p_array &l_in,
    const const_p_array &r_in, bool l_active, bool r_active);
}
#endif

