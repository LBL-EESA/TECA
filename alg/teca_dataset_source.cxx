#include "teca_dataset_source.h"
#include "teca_dataset.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;
using std::string;
using std::vector;

// --------------------------------------------------------------------------
teca_dataset_source::teca_dataset_source()
{
    this->set_number_of_input_connections(0);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_dataset_source::~teca_dataset_source()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_dataset_source::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    (void)prefix;
    (void)global_opts;
}

// --------------------------------------------------------------------------
void teca_dataset_source::set_properties(const std::string &prefix,
    variables_map &opts)
{
    (void)prefix;
    (void)opts;
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_dataset_source::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_dataset_source::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    return this->metadata;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_source::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_dataset_source::execute" << endl;
#endif
    (void)port;
    (void)input_data;
    (void)request;

    return this->dataset;
}
