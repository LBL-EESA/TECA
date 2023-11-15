#include "teca_dataset_source.h"
#include "teca_dataset.h"
#include "teca_table.h"
#include "teca_mesh.h"
#include "teca_metadata_util.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

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
    std::cerr << teca_parallel_id()
        << "teca_dataset_source::get_output_metadata" << std::endl;
#endif
    (void)port;
    (void)input_md;

    // let the user hanlde pipeline mechanics
    if (!this->metadata.empty())
        return this->metadata;

    unsigned int n_datasets = this->datasets.size();

    // handle pipeline mechanics for them
    teca_metadata omd;
    omd.set("index_initializer_key", std::string("num_datasets"));
    omd.set("num_datasets", n_datasets);
    omd.set("index_request_key", std::string("dataset_id"));

    // report the variables

    const_p_teca_mesh mesh = std::dynamic_pointer_cast
        <const teca_mesh>(n_datasets ? this->datasets[0] : nullptr);

    const_p_teca_table tab = std::dynamic_pointer_cast
        <const teca_table>(n_datasets ? this->datasets[0] : nullptr);

    if (mesh)
    {
        // report point centered arrays of the mesh
        std::vector<std::string> vars;
        unsigned int n_arrays = mesh->get_point_arrays()->size();
        for (unsigned int i = 0; i < n_arrays; ++i)
        {
            vars.push_back(mesh->get_point_arrays()->get_name(i));
        }
        omd.set("variables", vars);
    }
    else if (tab)
    {
        // report columns of the table
        std::vector<std::string> vars;
        unsigned int n_cols = tab->get_number_of_columns();
        for (unsigned int i = 0; i < n_cols; ++i)
        {
            vars.push_back(tab->get_column_name(i));
        }
        omd.set("variables", vars);
    }

    return omd;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_dataset_source::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_dataset_source::execute" << std::endl;
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
