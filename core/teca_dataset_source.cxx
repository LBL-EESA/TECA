#include "teca_dataset_source.h"
#include "teca_dataset.h"
#include "teca_metadata_util.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

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

    // let the user hanlde pipeline mechanics
    if (!this->metadata.empty())
        return this->metadata;

    // handle pipeline mechanics for them
    teca_metadata omd;
    omd.set("index_initializer_key", std::string("num_datasets"));
    omd.set("num_datasets", this->datasets.size());
    omd.set("index_request_key", std::string("dataset_id"));

    return omd;
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

    std::string request_key;
    unsigned long index = 0;
    if (teca_metadata_util::get_requested_index(request, request_key, index))
    {
        TECA_FATAL_ERROR("Failed to determine the requested index")
        return nullptr;
    }

    unsigned long num_datasets = this->datasets.size();
    if (index >= num_datasets)
    {
        TECA_FATAL_ERROR("No " << request_key << " index " << index << " in collection of "
            << num_datasets << " source datasets")
        return nullptr;
    }

    // serve it up
    p_teca_dataset ds = this->datasets[index];
    ds->set_request_index(request_key, index);

    return ds;
}
