#include "teca_apply_binary_mask.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <set>

#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::vector;
using std::set;
using std::cerr;
using std::endl;
using std::string;

namespace internal
{
template <typename mask_t, typename var_t>
void apply_mask(var_t * __restrict__ mask_output, const mask_t * __restrict__
mask_variable, const var_t * __restrict__ input_variable, unsigned long n)
{
    for (size_t i = 0; i < n; ++i)
    {
        mask_t m = mask_variable[i];
        var_t v = input_variable[i];
        mask_output[i] = m*v;
    }
}
};

//#define TECA_DEBUG// --------------------------------------------------------------------------
std::string teca_apply_binary_mask::get_output_variable_name(std::string input_var){
    return this->output_var_prefix + input_var;
}

// --------------------------------------------------------------------------
teca_apply_binary_mask::teca_apply_binary_mask() : 
    mask_variable(""), output_var_prefix("masked_")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_apply_binary_mask::~teca_apply_binary_mask()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_apply_binary_mask::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_apply_binary_mask":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, input_variables,
            "the input variables")
        TECA_POPTS_GET(std::string, prefix, mask_variable,
            "the name of the variable containing the mask array")
        TECA_POPTS_GET(std::string, prefix, output_var_prefix,
            "the prefix to apply to masked input variable names")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_apply_binary_mask::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, input_variables)
    TECA_POPTS_SET(opts, std::string, prefix, mask_variable)
    TECA_POPTS_SET(opts, std::string, prefix, output_var_prefix)
}
#endif
// --------------------------------------------------------------------------
teca_metadata teca_apply_binary_mask::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_apply_binary_mask::get_output_metadata" << endl;
#endif
    (void)port;

    if (this->input_variables.empty())
    {
        TECA_WARNING("The list of input variables was not set")
    }

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    // get the attributes
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    // construct the list of output variable names
    for (auto& input_var : input_variables){
        std::string output_var = this->get_output_variable_name(input_var);

        // add the varible to the list of output variables
        out_md.append("variables", output_var);

        // insert attributes to enable this variable to be written by the CF writer
        teca_metadata input_atts;
        if (attributes.get(input_var, input_atts))
        {
            TECA_WARNING("Failed to get attributes for \"" << input_var
                << "\". Writing the result will not be possible")
        }
        else
        {
            // copy the attributes from the input. this will capture the
            // data type, size, units, etc.
            teca_array_attributes output_atts(input_atts);

            // update description.
            output_atts.description = 
                std::string("masked/weighted by `" + this->mask_variable + "`");
            
            attributes.set(output_var, (teca_metadata)output_atts);
            out_md.set("attributes", attributes);
        }

    }
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_apply_binary_mask::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_apply_binary_mask::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    vector<teca_metadata> up_reqs;

    // get the name of the array to request
    if (this->mask_variable.empty())
    {
        TECA_ERROR("A mask variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(this->mask_variable);

    // check that a prefix was given
    if (this->get_output_var_prefix().empty()){
        TECA_ERROR("A prefix for the output variables was not specified")
        return up_reqs;
    }

    for (auto& input_var : input_variables){
        // insert the needed variable
        arrays.insert(input_var);

        // intercept request for our output if the variable will have a new name
        if(this->get_output_variable_name(input_var) != input_var){
            arrays.erase(this->get_output_variable_name(input_var));
        }
    }
    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_apply_binary_mask::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_apply_binary_mask::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);
    if (!in_mesh)
    {
        TECA_ERROR("Failed to apply mask. Dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // create the output mesh, pass everything through
    // output arrays are added in the variable loop
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // check that a masking variable has been provided
    if (this->mask_variable.empty())
    {
        TECA_ERROR("A mask variable was not specified")
        return nullptr;
    }

    // get the mask array
    const_p_teca_variant_array mask_array
        = in_mesh->get_point_arrays()->get(this->mask_variable);
    if (!mask_array)
    {
        TECA_ERROR("masking array \"" << this->mask_variable
            << "\" requested but not present.")
        return nullptr;
    }

    // apply the mask
    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        mask_array.get(), _mask,

        // loop over input variables
        for (auto& input_var : input_variables){
            std::string output_var = this->get_output_variable_name(input_var);
            // get the input array
            const_p_teca_variant_array input_array
                = in_mesh->get_point_arrays()->get(input_var);
            if (!input_array)
            {
                TECA_ERROR("input array \"" << input_var
                    << "\" requested but not present.")
                return nullptr;
            }

            // allocate the output array
            size_t n = input_array->size();
            p_teca_variant_array output_array = input_array->new_instance();
            output_array->resize(n);

            // do the mask calculation
            NESTED_TEMPLATE_DISPATCH_FP(
                teca_variant_array_impl,
                output_array.get(), _var,

                internal::apply_mask(
                    dynamic_cast<TT_var*>(output_array.get())->get(),
                    static_cast<const TT_mask*>(mask_array.get())->get(),
                    static_cast<const TT_var*>(input_array.get())->get(),
                    n);
                )

            out_mesh->get_point_arrays()->append(
                output_var, output_array);
        }
    )


    return out_mesh;
}
