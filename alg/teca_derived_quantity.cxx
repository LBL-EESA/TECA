#include "teca_derived_quantity.h"
#include "teca_dataset.h"

#include <string>
#include <vector>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif


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

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_derived_quantity::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);
}
#endif

// --------------------------------------------------------------------------
void teca_derived_quantity::set_operation_name(const std::string &v)
{
    if (this->operation_name != v)
    {
        this->set_name(v);
        this->operation_name = v;
        this->set_modified();
    }
}

// --------------------------------------------------------------------------
int teca_derived_quantity::set_name(const std::string &name)
{
    if (snprintf(this->class_name, sizeof(this->class_name),
        "teca_derived_quantity(%s)", name.c_str()) >=
        static_cast<int>(sizeof(this->class_name)))
    {
        TECA_FATAL_ERROR("name is too long for the current buffer size "
            << sizeof(this->class_name))
        return -1;
    }
    return 0;
}

/*
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
*/

// --------------------------------------------------------------------------
teca_metadata teca_derived_quantity::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_derived_quantity::get_output_metadata" << std::endl;
#endif
    (void)port;

    if (this->derived_variables.empty())
    {
        TECA_FATAL_ERROR("The name of the derived variable was not provided")
        return {};
    }

    teca_metadata out_md(input_md[0]);

    // get the array attributes collection
    teca_metadata atts;
    out_md.get("attributes", atts);

    unsigned int n_vars = this->derived_variables.size();
    for (unsigned int i = 0; i < n_vars; ++i)
    {
        // report the arrays we will generate
        out_md.append("variables", this->derived_variables[i]);

        // add the array attributes
        if (i < this->derived_variable_attributes.size())
        {
            atts.append(this->derived_variables[i],
                (teca_metadata)this->derived_variable_attributes[i]);
        }
        else
        {
            TECA_WARNING("No attributes were provided for "
                << this->derived_variables[i])
        }
    }

    // update the report
    out_md.set("attributes", atts);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_derived_quantity::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_derived_quantity::get_upstream_request" << std::endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    // intercept request for our output
    unsigned int n_vars = this->derived_variables.size();
    for (unsigned int i = 0; i < n_vars; ++i)
    {
        arrays.erase(this->derived_variables[i]);
    }

    // request the arrays we need
    size_t n = this->dependent_variables.size();
    for (size_t i = 0; i < n; ++i)
        arrays.insert(this->dependent_variables[i]);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}
