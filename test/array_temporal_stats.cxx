#include "array_temporal_stats.h"
#include "array.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace {
// helper to compute stats on the given array
int compute_stats(p_array input, p_array stats) noexcept
{
    double &min = (*stats)[0];
    double &max = (*stats)[1];
    double &avg = (*stats)[2];

    std::for_each(
        input->get_data().cbegin(),
        input->get_data().cend(),
        [&min, &avg, &max] (double v)
        {
            min = min > v ? v : min;
            max = max < v ? v : max;
            avg += v;
        });

    avg /= input->size();

    return 0;
}
};

// --------------------------------------------------------------------------
std::vector<teca_meta_data> array_temporal_stats::initialize_upstream_request(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md,
    const teca_meta_data &request)
{
    cerr << "array_temporal_stats::initialize_upstream_request" << endl;

    vector<teca_meta_data> up_reqs(1);
    up_reqs[0].set_prop("array_name", this->array_name);

    return up_reqs;
}

// --------------------------------------------------------------------------
teca_meta_data array_temporal_stats::initialize_output_meta_data(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md)
{
    cerr << "array_temporal_stats::intialize_output_meta_data" << endl;

    teca_meta_data output_md(input_md[0]);
    output_md.set_prop("array_name", this->array_name + "_stats");

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset array_temporal_stats::reduce(
    const p_teca_dataset &left,
    const p_teca_dataset &right)
{
    cerr << "array_temporal_stats::reduce" << endl;

    // validate inputs
    p_array l_in = std::dynamic_pointer_cast<array>(left);
    if (!l_in)
    {
        TECA_ERROR("left input is not an array")
        return p_teca_dataset();
    }

    p_array r_in = std::dynamic_pointer_cast<array>(right);
    if (!r_in)
    {
        TECA_ERROR("right input is not an array")
        return p_teca_dataset();
    }

    p_array stats = array::New();
    string active_array = this->array_name + "_stats";
    bool l_active = l_in->get_name() == active_array;
    bool r_active = r_in->get_name() == active_array;
    if (l_active && r_active)
    {
        (*stats)[0] = (*l_in)[0] < (*r_in)[0] ? (*l_in)[0] : (*r_in)[0];
        (*stats)[1] = (*l_in)[1] > (*r_in)[1] ? (*l_in)[1] : (*r_in)[1];
        (*stats)[2] = ((*l_in)[2] + (*r_in)[2])/2.0;
    }
    else
    if (l_active)
    {
        stats->copy_data(l_in);
        stats->copy_structure(l_in);
        ::compute_stats(r_in, stats);
    }
    else
    if (r_active)
    {
        stats->copy_data(r_in);
        stats->copy_structure(r_in);
        ::compute_stats(l_in, stats);
    }
    else
    {
        stats->set_name(this->array_name + "_stats");
        stats->resize(3);

        (*stats)[0] = std::numeric_limits<double>::max();
        (*stats)[1] = -std::numeric_limits<double>::max();
        (*stats)[2] = 0.0;

        ::compute_stats(l_in, stats);
        ::compute_stats(r_in, stats);
    }

    return stats;
}
