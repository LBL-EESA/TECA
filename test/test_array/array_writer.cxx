#include "array_writer.h"

#include "array.h"

#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
array_writer::array_writer()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
array_writer::~array_writer()
{}

// --------------------------------------------------------------------------
const_p_teca_dataset array_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) request;

    const_p_array a_in = std::dynamic_pointer_cast<const array>(input_data[0]);
    if (!a_in)
        return p_teca_dataset();

    ostringstream oss;
    a_in->to_stream(oss);

    cerr << teca_parallel_id() << "array_writer::execute " << oss.str() << endl;

    return p_teca_dataset();
}
