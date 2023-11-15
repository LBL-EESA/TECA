#include "array_temporal_stats_internals.h"

namespace array_temporal_stats_internals
{
namespace cpu
{
// **************************************************************************
p_array allocate_and_initialize_stats(const std::string &array_name)
{
    // imnportant! results are always alollocated on the CPU
    p_array results = array::new_host_accessible();

    results->set_name(array_name + "_stats");
    results->resize(4);

    array_temporal_stats_internals::cpu::initialize_stats(results->data());

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

    results = array::new_host_accessible();
    results->resize(4);

    // cases:
    if (l_active && r_active)
    {
        // both left and right contain new data
        // compute stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        p_array res_l = array::new_host_accessible();
        res_l->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            res_l->data(), pl_in.get(), l_in->size());

        // compute stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        p_array res_r = array::new_host_accessible();
        res_r->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            res_r->data(), pr_in.get(), r_in->size());

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), res_l->data(), res_r->data());
    }
    else
    if (l_active)
    {
        // left contains new data, right contains result

        // compute stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        p_array res_l = array::new_host_accessible();
        res_l->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            res_l->data(), pl_in.get(), l_in->size());

        // existing stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        // reduce stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), res_l->data(), pr_in.get());
    }
    else
    if (r_active)
    {
        // right contains data, left contains result

        // compute stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        p_array res_r = array::new_host_accessible();
        res_r->resize(4);

        array_temporal_stats_internals::cpu::compute_stats(
            res_r->data(), pr_in.get(), r_in->size());

        // existing stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        // reduce stats from the left (always on CPU)
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), pl_in.get(), res_r->data());
    }
    else
    {
        // existing stats from left
        std::shared_ptr<const double> pl_in = l_in->get_host_accessible();

        // existing stats from right
        std::shared_ptr<const double> pr_in = r_in->get_host_accessible();

        // both left and right contain stats
        array_temporal_stats_internals::cpu::reduce_stats(
            results->data(), pl_in.get(), pr_in.get());
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
