#ifndef teca_binary_segmentation_internals_h
#define teca_binary_segmentation_internals_h

#include "teca_binary_segmentation.h"
#include "teca_variant_array.h"

#include <algorithm>

namespace teca_binary_segmentation_internals
{
namespace cpu
{
// predicate for indirect sort
template <typename data_t, typename index_t>
struct indirect_lt
{
    indirect_lt() : p_data(nullptr) {}
    indirect_lt(const data_t *pd) : p_data(pd) {}

    bool operator()(const index_t &a, const index_t &b)
    {
        return p_data[a] < p_data[b];
    }

    const data_t *p_data;
};

template <typename data_t, typename index_t>
struct indirect_gt
{
    indirect_gt() : p_data(nullptr) {}
    indirect_gt(const data_t *pd) : p_data(pd) {}

    bool operator()(const index_t &a, const index_t &b)
    {
        return p_data[a] > p_data[b];
    }

    const data_t *p_data;
};

// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void value_threshold(out_t *output, const in_t *input,
    size_t n_vals, in_t low, in_t high)
{
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low) && (input[i] <= high)) ? 1 : 0;
}

// Given a vector V of length N, the q-th percentile of V is the value q/100 of
// the way from the minimum to the maximum in a sorted copy of V.

// set locations in the output where the input array
// has values within the low high range.
template <typename in_t, typename out_t>
void percentile_threshold(out_t *output, const in_t *input,
    unsigned long n_vals, float q_low, float q_high)
{
    // allocate indices and initialize
    using index_t = unsigned long;
    index_t *ids = (index_t*)malloc(n_vals*sizeof(index_t));
    for (index_t i = 0; i < n_vals; ++i)
        ids[i] = i;

    // cut points are locations of values bounding desired percentiles in the
    // sorted data
    index_t n_vals_m1 = n_vals - 1;

    // low percentile is bound from below by value at low_cut
    double tmp = n_vals_m1 * (q_low/100.f);
    index_t low_cut = index_t(tmp);
    double t_low = tmp - low_cut;

    // high percentile is bound from above by value at high_cut+1
    tmp = n_vals_m1 * (q_high/100.f);
    index_t high_cut = index_t(tmp);
    double t_high = tmp - high_cut;

    // compute 4 indices needed for percentile calcultion
    index_t low_cut_p1 = low_cut+1;
    index_t high_cut_p1 = std::min(high_cut+1, n_vals_m1);
    index_t *ids_pn_vals = ids+n_vals;

    // use an indirect comparison that leaves the input data unmodified
    indirect_lt<in_t,index_t>  comp(input);

    // find 2 indices needed for low percentile calc
    std::nth_element(ids, ids+low_cut, ids_pn_vals, comp);
    double y0 = input[ids[low_cut]];

    std::nth_element(ids, ids+low_cut_p1, ids_pn_vals, comp);
    double y1 = input[ids[low_cut_p1]];

    // compute low percetile
    double low_percentile = (y1 - y0)*t_low + y0;

    // find 2 indices needed for the high percentile calc
    std::nth_element(ids, ids+high_cut, ids_pn_vals, comp);
    y0 = input[ids[high_cut]];

    std::nth_element(ids, ids+high_cut_p1, ids_pn_vals, comp);
    y1 = input[ids[high_cut_p1]];

    // compute high percentile
    double high_percentile = (y1 - y0)*t_high + y0;

    /*std::cerr << q_low << "th percentile is " <<  std::setprecision(10) << low_percentile << std::endl
        << q_high << "th percentile is " <<  std::setprecision(9) << high_percentile << std::endl;*/

    // apply thresholds
    for (size_t i = 0; i < n_vals; ++i)
        output[i] = ((input[i] >= low_percentile) && (input[i] <= high_percentile)) ? 1 : 0;

    free(ids);
}
}

// do the segmentation on the cpu
int cpu_dispatch(
    p_teca_variant_array &output_array,
    const const_p_teca_variant_array &input_array,
    int threshold_mode,
    double low, double high);

// do the segmentation on the gpu
int cuda_dispatch(int device_id,
    p_teca_variant_array &output_array,
    const const_p_teca_variant_array &input_array,
    int threshold_mode,
    double low, double high);

}
#endif
