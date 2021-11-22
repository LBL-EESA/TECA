#include "array_scalar_multiply.h"
#include "array_scalar_multiply_internals.h"

#include "array.h"

#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
array_scalar_multiply::array_scalar_multiply() : scalar(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
array_scalar_multiply::~array_scalar_multiply()
{}

// --------------------------------------------------------------------------
int array_scalar_multiply::get_active_array(
    const teca_metadata &input_md,
    std::string &active_array) const
{
    if (this->array_name.empty())
    {
        // by default process the first array found on the input
        if (!input_md.has("array_names"))
        {
            TECA_ERROR("no array specified and none found on the input")
            return -1;
        }

        vector<string> array_names;
        input_md.get("array_names", array_names);

        active_array = array_names[0];
    }
    else
    {
        // otherwise process the requested array
        active_array = this->array_name;
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata array_scalar_multiply::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_scalar_multiply::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata output_md(input_md[0]);

    // if the user has requested a specific array then
    // replace "array_names" in the output metadata.
    // otherwise pass through and rely on down stream
    // requests to select the array to process
    if (this->array_name.size())
        output_md.set("array_names", this->array_name);

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> array_scalar_multiply::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "array_scalar_multiply::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // get the active array from the incoming request
    string active_array;
    if (request.get("array_name", active_array))
    {
        TECA_ERROR("array_name is not set on incoming the request")
        return up_reqs;
    }

    teca_metadata up_req(request);
    up_req.set("array_name", active_array);

    up_reqs.push_back(up_req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset array_scalar_multiply::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)request;

    // get the input array
    const_p_array a_in
        = std::dynamic_pointer_cast<const array>(input_data[0]);

    if (!a_in)
    {
        TECA_ERROR("no array to process")
        return p_teca_dataset();
    }

    // do the calculation
    p_array a_out;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        if (array_scalar_multiply_internals::cuda_dispatch(
            device_id, a_out, a_in, this->scalar, a_in->size()))
        {
            TECA_ERROR("Failed to multiply by a scalar on the GPU")
            return nullptr;
        }
    }
    else
    {
#endif
        if (array_scalar_multiply_internals::cpu_dispatch(
            a_out, a_in, this->scalar, a_in->size()))
        {
            TECA_ERROR("Failed to multiply by a scalar on the CPU")
            return nullptr;
        }
#if defined(TECA_HAS_CUDA)
    }
#endif

    // pass metadata
    a_out->copy_metadata(a_in);

#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_scalar_multiply::execute a_out=[";
    a_out->to_stream(std::cerr);
    std::cerr << "]" << std::endl;
#endif

    return a_out;
}
