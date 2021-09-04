#include "array_temporal_stats_internals.h"
#include "array_util.h"

namespace array_temporal_stats_internals
{
namespace cpu
{
// **************************************************************************
p_array allocate_and_initialize_stats(const std::string &array_name)
{
    // imnportant! results are always alollocated on the CPU
    p_array results = array::new_cpu_accessible();

    results->set_name(array_name + "_stats");
    results->resize(4);

    array_temporal_stats_internals::cpu::initialize_stats(results->get());

    return results;
}
}

// **************************************************************************
int cpu_dispatch(p_array &results, const const_p_array &l_in,
    const const_p_array &r_in, bool l_active, bool r_active)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_temporal_stats_internals::cpu_dispatch" << std::endl;
#endif

    results = array::new_cpu_accessible();
    results->resize(4);

    // cases:
    if (l_active && r_active)
    {
        // both left and right contain new data
        // compute stats from left
        const_p_array tmp_l_in = array_util::cpu_accessible(l_in);

        p_array tmp_res_l = array::new_cpu_accessible();
        tmp_res_l->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            tmp_res_l->get(), tmp_l_in->get(), tmp_l_in->size());

        // compute stats from right
        const_p_array tmp_r_in = array_util::cpu_accessible(r_in);

        p_array tmp_res_r = array::new_cpu_accessible();
        tmp_res_r->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            tmp_res_r->get(), tmp_r_in->get(), tmp_r_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), tmp_res_l->get(), tmp_res_r->get());
    }
    else
    if (l_active)
    {
        // left contains new data, right contains result

        // compute stats from left
        const_p_array tmp_l_in = array_util::cpu_accessible(l_in);

        p_array tmp_res_l = array::new_cpu_accessible();
        tmp_res_l->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            tmp_res_l->get(), tmp_l_in->get(), tmp_l_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), tmp_res_l->get(), r_in->get());
    }
    else
    if (r_active)
    {
        // right contains data, left contains result

        // compute stats from right
        const_p_array tmp_r_in = array_util::cpu_accessible(r_in);

        p_array tmp_res_r = array::new_cpu_accessible();
        tmp_res_r->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            tmp_res_r->get(), tmp_r_in->get(), tmp_r_in->size());

        // reduce stats from the left (always on CPU)
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), l_in->get(), tmp_res_r->get());
    }
    else
    {
        // both left and right contain stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->get(), l_in->get(), r_in->get());
    }

    return 0;
}

#if !defined(TECA_HAS_CUDA)
// **************************************************************************
int cuda_dispatch(int device, p_array &result, const const_p_array &l_in,
    const const_p_array &r_in, bool l_active, bool r_active)
{
    (void) device;
    (void) result;
    (void) l_in;
    (void) r_in;
    (void) l_active;
    (void) r_active;

    TECA_ERROR("array_temporal_stats failed because CUDA is not available")

    return -1;
}
#endif
}
