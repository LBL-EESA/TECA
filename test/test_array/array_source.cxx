#include "array_source.h"
#include "array.h"

#include "array.h"

#include <iostream>
#include <sstream>
#include <algorithm>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
array_source::array_source() :
    array_size(0),
    number_of_timesteps(0),
    time_delta(0.01)
{
    this->set_number_of_input_connections(0);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
array_source::~array_source()
{}

// --------------------------------------------------------------------------
void array_source::set_number_of_arrays(unsigned int n)
{
    if (this->array_names.size() != n)
    {
        this->array_names.clear();
        for (unsigned int i = 0; i < n; ++i)
        {
            ostringstream oss;
            oss << "array_" << i;
            this->array_names.push_back(oss.str());
        }
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
teca_metadata array_source::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_source::get_output_metadata" << endl;
#endif
    (void) port;
    (void) input_md;

    teca_metadata output_md;

    // report time
    vector<double> time;
    for (unsigned int i = 0; i < this->number_of_timesteps; ++i)
        time.push_back(this->time_delta*(i+1));
    output_md.insert("time", time);

    // report time steps
    output_md.insert("number_of_time_steps", this->number_of_timesteps);

    // report array extents
    vector<size_t> extent = {0, this->array_size};
    output_md.insert("extent", extent);

    // report array names
    output_md.insert("array_names", this->array_names);

    return output_md;
}

// --------------------------------------------------------------------------
const_p_teca_dataset array_source::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) input_data;

    // get the time request
    unsigned long time_step;
    if (request.get("time_step", time_step))
    {
        TECA_ERROR("request is missing \"time_step\"")
        return nullptr;
    }
    double active_time = this->time_delta*(time_step + 1);

    // get array request
    string active_array;
    if (request.get("array_name", active_array))
    {
        TECA_ERROR("request is missing array_name")
        return nullptr;
    }

    vector<string>::iterator it
        = std::find(
            this->array_names.begin(),
            this->array_names.end(),
            active_array);

    if (it == this->array_names.end())
    {
        TECA_ERROR(
            << "invalid array \"" << active_array << "\" requested");
        return nullptr;
    }
    size_t array_id = it - this->array_names.begin();

    // get the extent request
    vector<size_t> active_extent = {0, this->array_size};
    request.get("extent", active_extent);

    // generate the a_out array
    p_array a_out = array::New();
    a_out->set_extent(active_extent);
    a_out->set_name(active_array);

    for (unsigned int i = active_extent[0]; i < active_extent[1]; ++i)
        a_out->append(array_id + active_time);

#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_source::execute array=" << active_array
        << " time_step=" << time_step << " time=" << active_time
        << " extent=[" << active_extent[0] << ", " << active_extent[1]
        << "]" << endl;
#endif

    return a_out;
}
