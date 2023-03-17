#include "teca_rename_variables.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_rename_variables::teca_rename_variables() :
    original_variable_names(), new_variable_names()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_rename_variables::~teca_rename_variables()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_rename_variables::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_rename_variables":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, original_variable_names,
            "Sets the list of original_variable_names to rename.")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, new_variable_names,
            "Sets the list of new names, one for each variable to rename.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_rename_variables::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, original_variable_names)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, new_variable_names)

}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_rename_variables::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_rename_variables::get_output_metadata" << endl;
#endif
    (void)port;

    // validate the user provided values.
    if (this->original_variable_names.size() != this->new_variable_names.size())
    {
        TECA_FATAL_ERROR("Each variable to rename must have a "
            " corresponding output_variable_name.")
        return teca_metadata();
    }

    teca_metadata out_md(input_md[0]);

    // update the list of original_variable_names to reflect the new names
    std::set<std::string> out_vars;
    if (out_md.get("variables", out_vars))
    {
        TECA_FATAL_ERROR("Failed to get the list of variables")
        return teca_metadata();
    }

    unsigned long n_vars = this->original_variable_names.size();
    for (unsigned long i = 0; i < n_vars; ++i)
    {
        std::set<std::string>::iterator it = out_vars.find(this->original_variable_names[i]);
        if (it == out_vars.end())
        {
            TECA_FATAL_ERROR("No such variable \"" << this->original_variable_names[i]
                << "\" to rename")
            return teca_metadata();
        }

        out_vars.erase(it);
        out_vars.insert(this->new_variable_names[i]);
    }

    out_md.set("variables", out_vars);

    // update the list of attributes to reflect the new names
    teca_metadata attributes;
    if (out_md.get("attributes", attributes))
    {
        TECA_FATAL_ERROR("Failed to get attributes")
        return teca_metadata();
    }

    for (unsigned long i = 0; i < n_vars; ++i)
    {
        const std::string &var_name = this->original_variable_names[i];

        teca_metadata atts;
        if (attributes.get(var_name, atts))
        {
            TECA_FATAL_ERROR("Failed to get attributes for \"" << var_name << "\"")
            return teca_metadata();
        }

        attributes.remove(var_name);

        attributes.set(this->new_variable_names[i], atts);
    }

    out_md.set("attributes", attributes);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_rename_variables::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream requirements.
    // replace renamed original_variable_names with their original name
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    unsigned long n_vars = this->new_variable_names.size();
    for (unsigned long i = 0; i < n_vars; ++i)
    {
        std::set<std::string>::iterator it = arrays.find(this->new_variable_names[i]);
        if (it != arrays.end())
        {
            arrays.erase(it);
            arrays.insert(this->original_variable_names[i]);
        }

    }

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_rename_variables::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_rename_variables::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("The input dataset is not a teca_mesh")
        return nullptr;
    }

    // create the output mesh, pass everything through.
    p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());


    // rename the arrays if they are found
    p_teca_array_collection arrays = out_mesh->get_point_arrays();

    unsigned long n_vars = this->original_variable_names.size();
    for (unsigned long i = 0; i < n_vars; ++i)
    {
        const std::string var_name = this->original_variable_names[i];

        p_teca_variant_array array = arrays->get(var_name);
        if (array)
        {
            arrays->remove(var_name);
            arrays->set(this->new_variable_names[i], array);
        }
    }

    return out_mesh;
}
