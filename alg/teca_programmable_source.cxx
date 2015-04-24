#include "teca_programmable_source.h"

#include "teca_dataset.h"

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_programmable_source::teca_programmable_source()
{
    this->set_number_of_input_connections(0);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_programmable_source::~teca_programmable_source()
{}

// --------------------------------------------------------------------------
teca_metadata teca_programmable_source::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_programmable_source::get_output_metadata" << endl;
#endif
    (void) port;
    (void) input_md;

    return this->report_function();
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_programmable_source::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_programmable_source::execute" << endl;
#endif
    (void) port;
    (void) input_data;

    return this->execute_function(request);
}
