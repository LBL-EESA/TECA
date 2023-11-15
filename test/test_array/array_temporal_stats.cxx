#include "array_temporal_stats.h"
#include "array_temporal_stats_internals.h"
#include "array.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;


// --------------------------------------------------------------------------
array_temporal_stats::array_temporal_stats()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
    this->set_thread_pool_size(-1);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> array_temporal_stats::initialize_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_temporal_stats::initialize_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs(1, request);
    up_reqs[0].set("array_name", this->array_name);

    return up_reqs;
}

// --------------------------------------------------------------------------
teca_metadata array_temporal_stats::initialize_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_temporal_stats::intialize_output_metadata" << endl;
#endif
    (void) port;

    teca_metadata output_md(input_md[0]);
    output_md.set("array_names", this->array_name + "_stats");

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset array_temporal_stats::reduce(int device_id,
    const const_p_teca_dataset &left, const const_p_teca_dataset &right)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_temporal_stats::reduce" << std::endl;
#endif
    (void) device_id;

    // validate inputs
    const_p_array l_in = std::dynamic_pointer_cast<const array>(left);
    if (!l_in)
    {
        TECA_ERROR("left input is not an array")
        return p_teca_dataset();
    }

    const_p_array r_in = std::dynamic_pointer_cast<const array>(right);
    if (!r_in)
    {
        TECA_ERROR("right input is not an array")
        return p_teca_dataset();
    }

    // do the calculation of min, max, sum, and count
    p_array a_out;

    // check for the input array name to process. if found then new stats
    // are computed, otherwise existing stats are reduced.
    bool l_active = l_in->get_name() == this->array_name;
    bool r_active = r_in->get_name() == this->array_name;

#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        if (array_temporal_stats_internals::cuda_dispatch(
            device_id, a_out, l_in, r_in, l_active, r_active))
        {
            TECA_ERROR("Failed to compute stats on the GPU")
            return nullptr;
        }
    }
    else
    {
#endif
        if (array_temporal_stats_internals::cpu_dispatch(
            a_out, l_in, r_in, l_active, r_active))
        {
            TECA_ERROR("Failed to compute stats on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // pass metadata
    a_out->set_name(this->array_name + "_stats");

    return a_out;
}

// --------------------------------------------------------------------------
p_teca_dataset array_temporal_stats::finalize(int device_id,
    const const_p_teca_dataset &ds)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_temporal_stats::finalize" << std::endl;
#endif
    (void) device_id;

    const_p_array a_in = std::dynamic_pointer_cast<const array>(ds);
    if (!a_in)
    {
        TECA_ERROR("not an array")
        return nullptr;
    }

    std::shared_ptr<const double> pa_in = a_in->get_host_accessible();

    p_array a_out = array::new_host_accessible();
    a_out->resize(3);
    a_out->set_name(this->array_name + "_stats");


    const double *pi = pa_in.get();
    double *po = a_out->data();

    po[0] = pi[0];          // min
    po[1] = pi[1];          // max
    po[2] = pi[2]/pi[3];    // average

    return a_out;
}
