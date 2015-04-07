#include "cf_reader_driver.h"

#include <iostream>
#include <sstream>

using std::vector;
using std::string;
using std::ostringstream;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
cf_reader_driver::cf_reader_driver()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
cf_reader_driver::~cf_reader_driver()
{}

// --------------------------------------------------------------------------
teca_metadata cf_reader_driver::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "cf_reader_driver::get_output_metadata" << endl
        << "input_md = ";
    input_md[0].to_stream(cerr);
    cerr << endl;
#endif
    (void)port;
    teca_metadata output_md(input_md[0]);
    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> cf_reader_driver::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "cf_reader_driver::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;
    return up_reqs;
}

// --------------------------------------------------------------------------
p_teca_dataset cf_reader_driver::execute(
    unsigned int port,
    const std::vector<p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifndef TECA_NDEBUG
    cerr << teca_parallel_id()
        << "cf_reader_driver::execute" << endl;
#endif
    (void)port;
    (void)request;
    return nullptr;
}
