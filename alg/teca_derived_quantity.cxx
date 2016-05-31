#include "teca_derived_quantity.h"
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
teca_derived_quantity::teca_derived_quantity() :
    operation_name("teca_derived_quantity")
{}

// --------------------------------------------------------------------------
teca_derived_quantity::~teca_derived_quantity()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_derived_quantity::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_derived_quantity":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::vector<std::string>, prefix, dependent_variables,
            "list of arrays needed to compute the derived quantity")
        TECA_POPTS_GET(std::string, prefix, derived_variable,
            "name of the derived quantity")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_derived_quantity::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, dependent_variables)
    TECA_POPTS_SET(opts, std::string, prefix, derived_variable)
}
#endif

// --------------------------------------------------------------------------
void teca_derived_quantity::get_dependent_variables(
    const teca_metadata &request, std::vector<std::string> &dep_vars)
{
    dep_vars = this->dependent_variables;

    if (dep_vars.empty())
    {
        std::string key = this->operation_name + "::dependent_variables";
        if (request.has(key))
            request.get(key, dep_vars);
    }
}

// --------------------------------------------------------------------------
std::string teca_derived_quantity::get_derived_variable(
    const teca_metadata &request)
{
    std::string derived_var = this->derived_variable;

    if (derived_var.empty())
    {
        std::string key = this->operation_name + "::derived_variable";
        if (request.has(key))
            request.get(key, derived_var);
        else
            derived_var = "derived_quantity";
    }

    return derived_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_derived_quantity::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_derived_quantity::get_output_metadata" << endl;
#endif
    (void)port;

    // report the arrays we will generate
    teca_metadata out_md(input_md[0]);

    out_md.append("variables", this->derived_variable);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_derived_quantity::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_derived_quantity::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    // intercept request for our output
    arrays.erase(this->get_derived_variable(request));

    // get the names of the arrays we need to request
    std::vector<std::string> dep_vars;
    this->get_dependent_variables(req, dep_vars);

    size_t n = dependent_variables.size();
    for (size_t i = 0; i < n; ++i)
        arrays.insert(dep_vars[i]);

    req.insert("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}
