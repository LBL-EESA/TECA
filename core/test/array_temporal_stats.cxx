#include "array_temporal_stats.h"
#include "array.h"

#include <iostream>
#include <limits>

using std::cerr;
using std::endl;
using std::vector;
using std::string;


// --------------------------------------------------------------------------
array_temporal_stats::array_temporal_stats()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
p_array array_temporal_stats::new_stats_array(
    const_p_array l_input,
    const_p_array r_input)
{
    p_array stats = this->new_stats_array();

    stats->get(0) = l_input->get(0) < r_input->get(0) ? l_input->get(0) : r_input->get(0);
    stats->get(1) = l_input->get(1) > r_input->get(1) ? l_input->get(1) : r_input->get(1);
    stats->get(2) = (l_input->get(2) + r_input->get(2))/2.0;

    return stats;
}

// --------------------------------------------------------------------------
p_array array_temporal_stats::new_stats_array(const_p_array input)
{
    p_array stats = this->new_stats_array();

    double &min = stats->get(0);
    double &max = stats->get(1);
    double &avg = stats->get(2);

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

    return stats;
}

// --------------------------------------------------------------------------
p_array array_temporal_stats::new_stats_array()
{
    p_array stats = array::New();

    stats->set_name(array_name + "_stats");
    stats->resize(3);

    stats->get(0) = std::numeric_limits<double>::max();
    stats->get(1) = -std::numeric_limits<double>::max();
    stats->get(2) = 0.0;

    return stats;
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

    vector<teca_metadata> up_reqs(1, request);
    up_reqs[0].insert("array_name", this->array_name);

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
    output_md.insert("array_names", this->array_name + "_stats");

    return output_md;
}

// --------------------------------------------------------------------------
p_teca_dataset array_temporal_stats::reduce(
    const const_p_teca_dataset &left,
    const const_p_teca_dataset &right)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_temporal_stats::reduce" << endl;
#endif

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

    p_array stats;

    bool l_active = l_in->get_name() == this->array_name;
    bool r_active = r_in->get_name() == this->array_name;
    if (l_active && r_active)
    {
        // bopth left and right contain data
        p_array l_stats = this->new_stats_array(l_in);
        p_array r_stats = this->new_stats_array(r_in);
        stats = this->new_stats_array(l_stats, r_stats);
    }
    else
    if (l_active)
    {
        // left contains data, right contains stats
        p_array l_stats = this->new_stats_array(l_in);
        stats = this->new_stats_array(l_stats, r_in);
    }
    else
    if (r_active)
    {
        // right contains data, left contains stats
        p_array r_stats = this->new_stats_array(r_in);
        stats = this->new_stats_array(l_in, r_stats);
    }
    else
    {
        // both left and right contains stats
        stats = this->new_stats_array(l_in, r_in);
    }

    return stats;
}
