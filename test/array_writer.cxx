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
    this->set_number_of_inputs(1);
    this->set_number_of_outputs(1);
}

// --------------------------------------------------------------------------
array_writer::~array_writer()
{}

// --------------------------------------------------------------------------
p_teca_dataset array_writer::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_meta_data &request)
{
    p_array a_in = std::dynamic_pointer_cast<array>(input_data[0]);
    if (!a_in)
    {
        TECA_ERROR("no array to process")
        return p_teca_dataset();
    }

    cerr << "array_writer::execute array=" << a_in->get_name() << " extent=["
        << a_in->get_extent()[0] << ", " << a_in->get_extent()[1] << "] values=[";

    size_t n_elem = a_in->size();
    for (size_t i = 0; i < n_elem; ++i)
        cerr << " " << (*a_in)[i];

    cerr << " ]" << endl << endl;

    return p_teca_dataset();
}
