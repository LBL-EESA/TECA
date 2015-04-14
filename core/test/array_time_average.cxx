#include "array_time_average.h"

#include "array.h"

#include <algorithm>
#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
array_time_average::array_time_average() : filter_width(3)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
array_time_average::~array_time_average()
{}

// --------------------------------------------------------------------------
int array_time_average::get_active_times(
    const teca_metadata &input_md,
    const teca_metadata &request,
    double &current_time,
    std::vector<double> &active_times) const
{
    // get times available
    vector<double> time;
    if (input_md.get("time", time))
    {
        TECA_ERROR("time metadata not found")
        return -1;
    }

    // get the requested time
    request.get("time", current_time);

    vector<double>::iterator it = std::find(time.begin(), time.end(), current_time);
    if (it == time.end())
    {
        TECA_ERROR("invalid time request")
        return -2;
    }

    // request times centered about the current time
    int idx_0 = it - time.begin() - this->filter_width/2;
    int idx_1 = std::min((unsigned int)time.size(), idx_0 + this->filter_width);
    idx_0 = std::max(0, idx_0);

    for (int idx = idx_0; idx < idx_1; ++idx)
        active_times.push_back(time[idx]);

    return 0;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> array_time_average::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void) port;

    vector<teca_metadata> up_reqs;

    // get the active array from the incoming request
    string active_array;
    if (request.get("array_name", active_array))
    {
        TECA_ERROR("array_name is not set on incoming the request")
        return up_reqs;
    }

    // get the time values required to compute the average
    // centered on the requested time
    double current_time;
    vector<double> active_times;
    if (this->get_active_times(input_md[0], request, current_time, active_times))
    {
        TECA_ERROR("failed to get active times")
        return up_reqs;
    }

    // make a request for each time that will be used in the
    // average
    size_t n = active_times.size();
    for (size_t i = 0; i < n; ++i)
    {
        teca_metadata up_req(request);
        up_req.insert("array_name", active_array);
        up_req.insert("time", active_times[i]);
        up_reqs.push_back(up_req);
    }

#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_time_average::get_upstream_request array="
        << active_array << " time=" << current_time << " times=[ ";
    for (size_t i = 0; i < n; ++i)
        cerr << active_times[i] << " ";
    cerr << "]" << endl;
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
p_teca_dataset array_time_average::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_time_average::execute" << endl;
#endif
    (void) port;
    (void) request;

    p_array a_out = array::New();
    p_array a_in = std::dynamic_pointer_cast<array>(input_data[0]);
    if (a_in)
    {
        a_out->copy_metadata(a_in);
    }
    size_t n_elem = a_out->size();

    size_t n_times = input_data.size();
    for (size_t i = 0; i < n_times; ++i)
    {
        p_array a_in = std::dynamic_pointer_cast<array>(input_data[i]);
        if (!a_in)
        {
            TECA_ERROR("array " << i << " is invalid")
            return p_teca_dataset();
        }

        vector<double>::iterator out_it = a_out->get_data().begin();
        vector<double>::iterator in_it = a_in->get_data().begin();
        for (size_t j = 0; j < n_elem; ++j, ++out_it, ++in_it)
        {
            *out_it += *in_it;
        }
    }
    vector<double>::iterator out_it = a_out->get_data().begin();
    for (size_t j = 0; j < n_elem; ++j, ++out_it)
    {
        *out_it /= n_times;
    }

    return a_out;
}
