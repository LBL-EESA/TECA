#include "teca_table_join.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_table_join::teca_table_join()
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_join::~teca_table_join()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_join::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_join":prefix));

    /*opts.add_options()
        TECA_POPTS_GET(std::string, prefix, component_0_variable,
            "array containg the first component")
        TECA_POPTS_GET(std::string, prefix, component_1_variable,
            "array containg the second component")
        TECA_POPTS_GET(std::string, prefix, component_2_variable,
            "array containg the third component")
        TECA_POPTS_GET(std::string, prefix, table_join_variable,
            "array to store the computed norm in")
        ;*/

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_join::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    /*TECA_POPTS_SET(opts, std::string, prefix, component_0_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_1_variable)
    TECA_POPTS_SET(opts, std::string, prefix, component_2_variable)
    TECA_POPTS_SET(opts, std::string, prefix, table_join_variable)*/
}
#endif


// --------------------------------------------------------------------------
teca_metadata teca_table_join::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_table_join::get_output_metadata" << std::endl;
#endif
    (void)port;

    // start with a copy of metadata from the target
    unsigned int md_target = 0;
    teca_metadata output_md(input_md[md_target]);

    // get target metadata
    std::set<std::string> target_vars;
    input_md[md_target].get("variables", target_vars);

    teca_metadata target_atts;
    input_md[md_target].get("attributes", target_atts);

    // work with each source
    unsigned int n_in = this->get_number_of_input_connections();
    for (unsigned int i = 1; i < n_in; ++i)
    {
        unsigned int md_src = i;

        // get source metadata
        std::vector<std::string> source_vars;
        input_md[md_src].get("variables", source_vars);

        teca_metadata source_atts;
        input_md[md_src].get("attributes", source_atts);

        // merge metadata from source and target variables should be unique
        // lists.  attributes are indexed by variable names in the case of
        // collisions, the target variable is kept, the source variable is
        // ignored
        size_t n_source_vars = source_vars.size();
        for (size_t i = 0; i < n_source_vars; ++i)
        {
            const std::string &src_var = source_vars[i];

            auto [it, ins] = target_vars.insert(src_var);

            if (ins)
            {
                teca_metadata atts;
                source_atts.get(src_var, atts);
                target_atts.set(src_var, atts);
            }

        }
    }

    // update with merged lists
    output_md.set("variables", target_vars);
    output_md.set("attributes", target_atts);

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_table_join::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    // route requests for arrays to either target or the input.
    // if the array exists in both then it is take from the target

    // start by duplicating the request for each input
    unsigned int n_in = this->get_number_of_input_connections();
    std::vector<teca_metadata> up_reqs(n_in, request);

    // get input metadata
    std::vector<std::set<std::string>> input_vars(n_in);
    for (unsigned int i = 0; i < n_in; ++i)
    {
        input_md[i].get("variables", input_vars[i]);
    }

    // get the requested arrays
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    // route the request for each array to the most appropriate input. in the
    // case of inputs providing the same array the request is sent to the lower
    // input.
    std::vector<std::set<std::string>> up_req_arrays(n_in);

    auto it = req_arrays.begin();
    auto end = req_arrays.end();
    for (; it != end; ++it)
    {
        // work with each input
        bool array_found = false;
        for (unsigned int i = 0; i < n_in; ++i)
        {
            // check if the i'th input has the array
            if (input_vars[i].count(*it))
            {
                // request from the i'th input
                up_req_arrays[i].insert(*it);
                array_found = true;
                break;
            }
        }

        // require that at least one input can provide the requested array
        if (!array_found)
        {
            TECA_FATAL_ERROR("\"" << *it << "\" was not found on any input")
            return {};
        }
    }

    // update the requests
    for (unsigned int i = 0; i < n_in; ++i)
        up_reqs[i].set("arrays", up_req_arrays[i]);

#ifdef TECA_DEBUG
    for (unsigned int i = 0; i < n_in; ++i)
    {
        std::cerr << "request[" << i << "] = ";
        up_reqs[i].to_stream(std::cerr);
        std::cerr << std::endl;
    }
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_join::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_table_join::execute" << std::endl;
#endif
    (void)port;

    unsigned int n_in = this->get_number_of_input_connections();

    p_teca_table in_target
        = std::dynamic_pointer_cast<teca_table>(
            std::const_pointer_cast<teca_dataset>(input_data[0]));

    if (!in_target)
    {
        TECA_FATAL_ERROR("no target table is "
            << (input_data[0] ? input_data[0]->get_class_name() : "empty"))
        return nullptr;
    }

    // create the output
    p_teca_table target = teca_table::New();
    target->shallow_copy(in_target);

    // get the attributes
    teca_metadata target_atts;
    target->get_attributes(target_atts);

    unsigned long n_rows = target->get_number_of_rows();

    // get the list of arrays to move
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    auto it = req_arrays.begin();
    auto end = req_arrays.end();

    for (; it != end; ++it)
    {
        if (!target->has_column(*it))
        {
            // join columns from the other tables
            bool col_found = false;

            for (unsigned int i = 1; i < n_in; ++i)
            {
                const_p_teca_table source
                    = std::dynamic_pointer_cast<const teca_table>
                        (input_data[i]);

                if (!source)
                {
                    TECA_FATAL_ERROR("source table " << i << " is "
                        << (source ? source->get_class_name() : "empty"))
                    return nullptr;
                }

                if (source->has_column(*it))
                {
                    p_teca_variant_array col
                        = std::const_pointer_cast<teca_variant_array>
                            (source->get_column(*it));

                    // check for consistencey across tables to join
                    if (col->size() != n_rows)
                    {
                        TECA_ERROR("Column " << *it << " with " << col->size()
                            << " rows is inconsistent with a table with " << n_rows)
                    }

                    // pass the column
                    target->append_column(*it, col);

                    // pass the attributes
                    teca_metadata source_atts;
                    source->get_attributes(source_atts);

                    teca_metadata array_atts;
                    if (!source_atts.get(*it, array_atts))
                        target_atts.set(*it, array_atts);

                    col_found = true;
                    break;
                }
            }

            if (!col_found)
            {
                TECA_FATAL_ERROR("Array \"" << *it
                    << "\" is not present on any of the inputs")
                return nullptr;
            }
        }
    }

    // update the attributes
    target->set_attributes(target_atts);

    return target;
}
