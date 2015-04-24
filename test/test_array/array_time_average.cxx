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
        TECA_ERROR("request is missing \"array_name\"")
        return up_reqs;
    }

    // get the time values required to compute the average
    // centered on the requested time
    long active_step;
    if (request.get("time_step", active_step))
    {
        TECA_ERROR("request is missing \"time_step\"")
        return up_reqs;
    }

    long num_steps;
    if (input_md[0].get("number_of_time_steps", num_steps))
    {
        TECA_ERROR("input is missing \"number_of_time_steps\"")
        return up_reqs;
    }

    long first = active_step - this->filter_width/2;
    long last = active_step + this->filter_width/2;
    for (long i = first; i <= last; ++i)
    {
        // make a request for each time that will be used in the
        // average
        if ((i >= 0) && (i < num_steps))
        {
            teca_metadata up_req(request);
            up_req.insert("array_name", active_array);
            up_req.insert("time_step", i);
            up_reqs.push_back(up_req);
        }
    }

#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_time_average::get_upstream_request array="
        << active_array << " active_step=" << active_step << " steps=[ ";
    for (long i = first; i <= last; ++i)
        if ((i >= 0) && (i < num_steps))
            cerr << i << " ";
    cerr << "]" << endl;
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset array_time_average::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_time_average::execute" << endl;
#endif
    (void) port;
    (void) request;

    p_array a_out = array::New();
    const_p_array a_in = std::dynamic_pointer_cast<const array>(input_data[0]);
    if (a_in)
    {
        a_out->copy_metadata(a_in);
    }
    size_t n_elem = a_out->size();

    size_t n_times = input_data.size();
    for (size_t i = 0; i < n_times; ++i)
    {
        const_p_array a_in = std::dynamic_pointer_cast<const array>(input_data[i]);
        if (!a_in)
        {
            TECA_ERROR("array " << i << " is invalid")
            return p_teca_dataset();
        }

        vector<double>::iterator out_it = a_out->get_data().begin();
        vector<double>::const_iterator in_it = a_in->get_data().cbegin();
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
