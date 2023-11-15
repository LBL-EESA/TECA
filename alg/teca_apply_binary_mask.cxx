#include "teca_apply_binary_mask.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_mpi_util.h"

#include <algorithm>
#include <iostream>
#include <set>

#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG

using std::cerr;
using std::endl;

using namespace teca_variant_array_util;

namespace internal
{
// output = mask*input
template <typename mask_t, typename var_t>
void apply_mask(var_t * __restrict__ output,
    const mask_t * __restrict__ mask,
    const var_t * __restrict__ input,
    unsigned long n)
{
    for (size_t i = 0; i < n; ++i)
        output[i] = mask[i]*input[i];
}
}

// --------------------------------------------------------------------------
teca_apply_binary_mask::teca_apply_binary_mask() :
    mask_variable(""), output_variable_prefix("masked_")
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
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_apply_binary_mask":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, masked_variables,
            "A list of variables to apply the mask to.")
        TECA_POPTS_GET(std::string, prefix, mask_variable,
            "The name of the variable containing the mask values.")
        TECA_POPTS_GET(std::string, prefix, output_variable_prefix,
            "A string prepended to the output variable names. If empty the"
            " input variables will be replaced by their masked results")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_apply_binary_mask::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, masked_variables)
    TECA_POPTS_SET(opts, std::string, prefix, mask_variable)
    TECA_POPTS_SET(opts, std::string, prefix, output_variable_prefix)
}
#endif

// --------------------------------------------------------------------------
std::string teca_apply_binary_mask::get_output_variable_name(std::string input_var)
{
    return this->output_variable_prefix + input_var;
}

// --------------------------------------------------------------------------
void teca_apply_binary_mask::get_output_variable_names(
    std::vector<std::string> &names)
{
    int n_inputs = this->masked_variables.size();
    for (int i = 0; i < n_inputs; ++i)
    {
        names.push_back(
            this->get_output_variable_name(this->masked_variables[i]));
    }
}

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

    // check that the input variables have been specified.
    // this is likely a user error.
    if (this->masked_variables.empty() &&
        teca_mpi_util::mpi_rank_0(this->get_communicator()))
    {
        TECA_WARNING("Nothing to do, masked_variables have not"
            " been specified.")
    }

    // add in the array we will generate
    teca_metadata out_md(input_md[0]);

    // get the attributes
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    // construct the list of output variable names
    for (auto& input_var : masked_variables)
    {
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

            // update description and long name
            output_atts.description = input_var +
                " multiplied by " + this->mask_variable;

            output_atts.long_name.clear();

            // update the array attributes
            attributes.set(output_var, (teca_metadata)output_atts);
        }

    }

    // update the attributes
    out_md.set("attributes", attributes);

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

    std::vector<teca_metadata> up_reqs;

    // get the name of the mask array
    if (this->mask_variable.empty())
    {
        TECA_FATAL_ERROR("A mask variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;

    if (req.has("arrays"))
        req.get("arrays", arrays);

    arrays.insert(this->mask_variable);

    // check that the input variables have been specified.
    // this is likely a user error.
    if (this->masked_variables.empty() &&
        teca_mpi_util::mpi_rank_0(this->get_communicator()))
    {
        TECA_WARNING("Nothing to do, masked_variables have not"
            " been specified.")
    }

    // request the arrays to mask
    for (auto& input_var : masked_variables)
    {
        // request the needed variable
        arrays.insert(input_var);

        // intercept request for our output if the variable will have a new name
        std::string out_var = this->get_output_variable_name(input_var);
        if (out_var != input_var)
        {
            arrays.erase(out_var);
        }
    }

    // update the list of arrays to request
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
    const_p_teca_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);
    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to apply mask. Dataset is not a teca_mesh")
        return nullptr;
    }

    // create the output mesh, pass everything through
    // masked arrays are added or replaced below
    p_teca_mesh out_mesh =
        std::static_pointer_cast<teca_mesh>
            (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());

    // check that a masking variable has been provided
    if (this->mask_variable.empty())
    {
        TECA_FATAL_ERROR("The mask_variable name was not specified")
        return nullptr;
    }

    // get the mask array
    const_p_teca_variant_array mask_array
        = in_mesh->get_point_arrays()->get(this->mask_variable);
    if (!mask_array)
    {
        TECA_FATAL_ERROR("The mask_variable \"" << this->mask_variable
            << "\" was requested but is not present in the input data.")
        return nullptr;
    }

    // apply the mask
    NESTED_VARIANT_ARRAY_DISPATCH(
        mask_array.get(), _MASK,

        auto [sp_mask, p_mask] = get_host_accessible<CTT_MASK>(mask_array);

        // loop over input variables
        for (auto& input_var : masked_variables)
        {
            std::string output_var = this->get_output_variable_name(input_var);

            // get the input array
            const_p_teca_variant_array input_array
                = in_mesh->get_point_arrays()->get(input_var);
            if (!input_array)
            {
                TECA_FATAL_ERROR("The masked_variable \"" << input_var
                    << "\" was requested but is not present in the input data.")
                return nullptr;
            }

            // allocate the output array
            size_t n = input_array->size();
            p_teca_variant_array output_array = input_array->new_instance(n);

            // do the mask calculation
            NESTED_VARIANT_ARRAY_DISPATCH(
                output_array.get(), _VAR,

                auto [sp_in, p_in] = get_host_accessible<CTT_VAR>(input_array);
                auto [p_out] = data<TT_VAR>(output_array);

                sync_host_access_any(mask_array, input_array);

                internal::apply_mask(p_out, p_mask, p_in, n);
                )

            out_mesh->get_point_arrays()->set(
                output_var, output_array);
        }
    )

    return out_mesh;
}
