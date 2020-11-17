#include "teca_dataset_capture.h"
#include "teca_dataset.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_dataset_capture::teca_dataset_capture()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_dataset_capture::~teca_dataset_capture()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_dataset_capture::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    (void)prefix;
    (void)global_opts;
}

// --------------------------------------------------------------------------
void teca_dataset_capture::set_properties(const std::string &prefix,
    variables_map &opts)
{
    (void)prefix;
    (void)opts;
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_capture::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_dataset_capture::execute" << endl;
#endif
    (void)port;
    (void)request;
    this->dataset = input_data[0];
    return this->dataset;
}
