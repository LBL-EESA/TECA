#include "array_source.h"
#include "array.h"
#include "array_source_internals.h"
#include "teca_config.h"
#include "teca_metadata_util.h"

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
    output_md.set("time", time);

    // report time steps
    output_md.set("number_of_time_steps", this->number_of_timesteps);

    // report array extents
    vector<size_t> extent = {0, this->array_size};
    output_md.set("extent", extent);

    // report array names
    output_md.set("array_names", this->array_names);

    // let the excutive know how to make requests
    output_md.set("index_initializer_key", std::string("number_of_time_steps"));
    output_md.set("index_request_key", std::string("time_step"));

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
    std::string request_key;
    unsigned long time_step = 0;
    if (teca_metadata_util::get_requested_index(request, request_key, time_step))
    {
        TECA_ERROR("Failed to determine the requested index")
        return nullptr;
    }

    double active_time = this->time_delta * (time_step + 1);

    // get array request
    string active_array;
    if (request.get("array_name", active_array))
    {
        TECA_ERROR("request is missing array_name")
        return nullptr;
    }

    vector<string>::iterator it = std::find(
        this->array_names.begin(), this->array_names.end(), active_array);

    if (it == this->array_names.end())
    {
        TECA_ERROR("invalid array \"" << active_array << "\" requested")
        return nullptr;
    }

    // get the index of the current array
    size_t array_id = it - this->array_names.begin();

    // get the extent request
    size_t active_extent[2] = {0, this->array_size};
    request.get("extent", active_extent);

    // create the output dataset
    p_array a_out;

    // intialize to this value
    double init_val = array_id + active_time;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (array_source_internals::cuda_dispatch(
            device_id, a_out, init_val, this->array_size))
        {
            TECA_ERROR("Failed to initialize the data on the GPU")
            return nullptr;
        }
    }
    else
    {
#endif
        if (array_source_internals::cpu_dispatch(
            a_out, init_val, this->array_size))
        {
            TECA_ERROR("Failed to initialize the data on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // pass metadata
    a_out->set_extent({active_extent[0], active_extent[1]});
    a_out->set_name(active_array);

#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_source::execute array=" << active_array
        << " time_step=" << time_step << " time=" << active_time
        << " extent=[" << active_extent[0] << ", " << active_extent[1]
        << "] a_out=[";
    a_out->to_stream(std::cerr);
    std::cerr << "]" << std::endl;
#endif

    return a_out;
}
