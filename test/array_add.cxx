#include "array_add.h"

#include "array.h"

#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
array_add::array_add()
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
array_add::~array_add()
{}

// --------------------------------------------------------------------------
int array_add::get_active_array(
    const std::string &user_array,
    const teca_meta_data &input_md,
    std::string &active_array) const
{
    if (user_array.empty())
    {
        // by default process the first array found on the input
        if (!input_md.has("array_names"))
        {
            TECA_ERROR("no array specified and none found on the input")
            return -1;
        }

        vector<string> array_names;
        input_md.get_prop("array_names", array_names);

        active_array = array_names[0];
    }
    else
    {
        // otherwise process the requested array
        active_array = user_array;
    }
    return 0;
}


// --------------------------------------------------------------------------
teca_meta_data array_add::get_output_meta_data(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md)
{
    cerr << "array_add::get_output_meta_data" << endl;

    teca_meta_data output_md(input_md[0]);

    // get the active arrays
    string active_array_1;
    string active_array_2;
    if ( this->get_active_array(this->array_1, input_md[0], active_array_1)
      || this->get_active_array(this->array_2, input_md[1], active_array_2) )
    {
        TECA_ERROR("failed to get the active arrays")
        return output_md;
    }

    // replace the "array_names" key in the meta data. down stream
    // filters only see what this filter generates.
    ostringstream oss;
    oss << active_array_1 << "_plus_" << active_array_2;
    output_md.set_prop("array_names", oss.str());

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_meta_data> array_add::get_upstream_request(
    unsigned int port,
    const std::vector<teca_meta_data> &input_md,
    const teca_meta_data &request)
{
    cerr << "array_add::get_upstream_request" << endl;

    vector<teca_meta_data> up_reqs(2);

    // get the active arrays
    string active_array_1;
    string active_array_2;
    if ( this->get_active_array(this->array_1, input_md[0], active_array_1)
      || this->get_active_array(this->array_2, input_md[1], active_array_2) )
    {
        TECA_ERROR("failed to get the active arrays")
        return up_reqs;
    }

    teca_meta_data up_req_1(request);
    up_req_1.set_prop("array_name", active_array_1);
    up_reqs[0] = up_req_1;

    teca_meta_data up_req_2(request);
    up_req_2.set_prop("array_name", active_array_2);
    up_reqs[1] = up_req_2;

    return up_reqs;
}

// --------------------------------------------------------------------------
p_teca_dataset array_add::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_meta_data &request)
{
    p_array a_out = array::New();

    // get the array on the two inputs
    p_array a_in_1 = std::dynamic_pointer_cast<array>(input_data[0]);
    p_array a_in_2 = std::dynamic_pointer_cast<array>(input_data[1]);
    if (!a_in_1 || !a_in_2)
    {
        TECA_ERROR("required inputs are not present")
        return a_out;
    }

    // get the output array name
    string active_array;
    if (request.get_prop("array_name", active_array))
    {
        TECA_ERROR("failed to get the active array")
        return a_out;
    }

    // compute the output
    a_out->copy_structure(a_in_1);
    a_out->set_name(active_array);

    size_t n_elem = a_out->size();
    for (size_t i = 0; i < n_elem; ++i)
        a_out->get(i) = a_in_1->get(i) + a_in_2->get(i);

    cerr << "array_add::execute " << active_array << endl;

    return a_out;
}
