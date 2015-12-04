#include "teca_programmable_algorithm.h"

#include "teca_dataset.h"

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_programmable_algorithm::teca_programmable_algorithm()
{
    this->set_number_of_input_connections(0);
    this->set_number_of_output_ports(0);

    // set default functions
    this->set_report_function(
        [](unsigned int, const std::vector<teca_metadata> &input_md)
            -> teca_metadata
        {
            // the default implementation passes meta data through
            if (input_md.size())
                return input_md[0];

            return teca_metadata();
        });

    this->set_request_function(
        [this](unsigned int, const std::vector<teca_metadata> &,
            const teca_metadata &request) -> std::vector<teca_metadata>
        {
            // default implementation forwards request upstream
            return std::vector<teca_metadata>(
                this->get_number_of_input_connections(), request);
        });

    this->set_execute_function(
        [] (unsigned int, const std::vector<const_p_teca_dataset> &,
            const teca_metadata &) -> const_p_teca_dataset
        {
            // default implementation does nothing
            return p_teca_dataset();
        });
}

// --------------------------------------------------------------------------
teca_programmable_algorithm::~teca_programmable_algorithm()
{}

// --------------------------------------------------------------------------
teca_metadata teca_programmable_algorithm::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_programmable_algorithm::get_output_metadata" << endl;
#endif

    return this->report_function(port, input_md);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_programmable_algorithm::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_programmable_algorithm::get_upstream_request" << endl;
#endif

    return this->request_function(port, input_md, request);
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_programmable_algorithm::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_programmable_algorithm::execute" << endl;
#endif

    return this->execute_function(port, input_data, request);
}
