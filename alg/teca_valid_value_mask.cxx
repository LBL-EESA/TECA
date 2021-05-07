#include "teca_valid_value_mask.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_coordinate_util.h"
#include "teca_mpi.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>
#include <limits>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

namespace
{
bool is_mask_array(const std::string &array)
{
    size_t n = array.size();
    size_t pos = n - 6;

    if ((n < 6) || (strncmp(array.c_str() + pos, "_valid", 6) != 0))
        return false;

    return true;
}
}

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_valid_value_mask::teca_valid_value_mask() :
    mask_arrays(), enable_valid_range(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_valid_value_mask::~teca_valid_value_mask()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_valid_value_mask::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_valid_value_mask":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>,
             prefix, mask_arrays,
            "A list of arrays to compute a mask for.")
        TECA_POPTS_GET(int, prefix, enable_valid_range,
            "If set non-zero vald_range, valid_min, and valid_max attributes"
            " would be used if there is no _FillValue attribute.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_valid_value_mask::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, mask_arrays)
    TECA_POPTS_SET(opts, int, prefix, enable_valid_range)
}
#endif


// --------------------------------------------------------------------------
teca_metadata teca_valid_value_mask::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_valid_value_mask::get_output_metadata" << std::endl;
#endif
    (void)port;

    // get the list of available variables and their attriibutes
    teca_metadata out_md(input_md[0]);

    std::vector<std::string> variables(this->mask_arrays);
    if (variables.empty() && out_md.get("variables", variables))
    {
        TECA_ERROR("Failed to get the list of variables")
        return teca_metadata();
    }

    teca_metadata attributes;
    if (out_md.get("attributes", attributes))
    {
        TECA_ERROR("Failed to get the array attributes")
        return teca_metadata();
    }

    // for each mask array we might generate, report that it is available and
    // supply attributes to enable the CF writer.
    size_t n_arrays = variables.size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        const std::string &array_name = variables[i];

        teca_metadata array_atts;
        if (attributes.get(array_name, array_atts))
        {
            // this could be reported as an error or a warning but unless this
            // becomes problematic quietly ignore it
            continue;
        }

        // get the centering and size from the array
        unsigned int centering = 0;
        array_atts.get("centering", centering);

        unsigned long size = 0;
        array_atts.get("size", size);

        // construct attributes
        teca_array_attributes mask_atts(
            teca_variant_array_code<char>::get(),
            centering, size, "none", "", "valid value mask");

        std::string mask_name = array_name + "_valid";

        // update attributes
        attributes.set(mask_name, (teca_metadata)mask_atts);

        // add to the list of available variables
        out_md.append("variables", mask_name);
    }

    // update the output metadata
    out_md.set("attributes", attributes);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_valid_value_mask::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_valid_value_mask::get_output_metadata" << std::endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    // get the requested arrays. pass up only those that we don't generate.
    std::vector<std::string> arrays;
    req.get("arrays", arrays);

    std::set<std::string> arrays_up;

    int n_arrays = arrays.size();
    for (int i = 0; i < n_arrays; ++i)
    {
        const std::string &array = arrays[i];
        if (::is_mask_array(array))
        {
            // remove _valid and request the base array
            arrays_up.insert(array.substr(0, array.size()-6));
        }
        else
        {
            // not ours, pass through
            arrays_up.insert(array);
        }
    }

    // request explcitly named arrays
    if (!this->mask_arrays.empty())
    {
        arrays_up.insert(this->mask_arrays.begin(),
            this->mask_arrays.end());
    }

    // update the request
    req.set("arrays", arrays_up);

    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_valid_value_mask::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_valid_value_mask::execute" << std::endl;
#endif
    (void)port;

        int rank = 0;
#if defined(TECA_HAS_MPI)
        MPI_Comm comm = this->get_communicator();
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
            MPI_Comm_rank(comm, &rank);
#endif

    // get the input mesh
    const_p_teca_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Empty input dataset or not a teca_mesh")
        return nullptr;
    }

    // allocate the output
    p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>
        (std::const_pointer_cast<teca_mesh>(in_mesh)->new_shallow_copy());

    // get the arrays to process
    std::vector<std::string> tmp = this->mask_arrays;
    if (tmp.empty())
    {
        request.get("arrays", tmp);
    }

    std::vector<std::string> arrays;
    int n_arrays = tmp.size();
    for (int i = 0; i < n_arrays; ++i)
    {
        const std::string &array = tmp[i];
        if (::is_mask_array(array))
        {
            arrays.push_back(array.substr(0, array.size()-6));
        }
    }

    // get the array attributes, the fill value controls will be found here
    teca_metadata &md = out_mesh->get_metadata();

    teca_metadata attributes;
    if (md.get("attributes", attributes))
    {
        TECA_ERROR("Failed to get the array attributes")
        return nullptr;
    }

    // for each array generate the mask
    n_arrays = arrays.size();
    for (int i = 0; i < n_arrays; ++i)
    {
        const std::string &array_name = arrays[i];

        // get the attributes
        teca_metadata array_atts;
        if (attributes.get(array_name, array_atts))
        {
            TECA_ERROR("The mask for array \"" << array_name
                << "\" not computed. The array has no attributes")
            return nullptr;
        }


        // get the centering
        unsigned int centering = 0;
        if (array_atts.get("centering", centering))
        {
            TECA_ERROR("Mask for array \"" << array_name << "\" not computed."
                " Attributes are missing centering metadata")
            return nullptr;
        }

        p_teca_array_collection arrays = out_mesh->get_arrays(centering);
        if (!arrays)
        {
            TECA_ERROR("Mask for array \"" << array_name << "\" not computed."
                " Failed to get the array collection with centering " << centering)
            return nullptr;
        }

        // get the input array
        p_teca_variant_array array = arrays->get(array_name);
        if (!array)
        {
            TECA_ERROR("Mask for array \"" << array_name << "\" not computed."
                " No array named \"" << array_name << "\"")
            return nullptr;
        }


        TEMPLATE_DISPATCH(teca_variant_array_impl,
            array.get(),

            // look for a _FillValue
            bool have_fill_value = false;

            NT fill_value = std::numeric_limits<NT>::max();

            have_fill_value = ((array_atts.get("_FillValue", fill_value) == 0) ||
                (array_atts.get("missing_value", fill_value) == 0));

            // look for some combination of valid rnage attributes.
            bool have_valid_range = false;
            bool have_valid_min = false;
            bool have_valid_max = false;

            NT valid_range[2];
            valid_range[0] = std::numeric_limits<NT>::lowest();
            valid_range[1] = std::numeric_limits<NT>::max();

            if (this->enable_valid_range)
            {
                have_valid_range = !have_fill_value &&
                    (array_atts.get("valid_range", valid_range, 2) == 0);

                have_valid_min = !have_fill_value && !have_valid_range &&
                    (array_atts.get("valid_min", valid_range[0]) == 0);

                have_valid_max = !have_fill_value && !have_valid_range &&
                    (array_atts.get("valid_max", valid_range[1]) == 0);
            }

            // get a pointer to the values
            const NT *p_array = static_cast<TT*>(array.get())->get();
            size_t n_elem = array->size();

            p_teca_char_array mask;

            if (have_fill_value)
            {
                // allocate and compute the mask
                mask = teca_char_array::New(n_elem);
                char *p_mask = mask->get();
                for (size_t i = 0; i < n_elem; ++i)
                {
                    p_mask[i] = teca_coordinate_util::equal(p_array[i], fill_value) ? 0 : 1;
                }

                if (this->verbose && (rank == 0))
                {
                    TECA_STATUS("Mask for array \""
                        << array_name << "\" will be generated using _FillValue="
                        << fill_value)
                }
            }
            else if (have_valid_min || have_valid_max || have_valid_range)
            {
                // allocate and compute the mask
                mask = teca_char_array::New(n_elem);
                char *p_mask = mask->get();
                for (size_t i = 0; i < n_elem; ++i)
                {
                    NT val = p_array[i];
                    p_mask[i] = ((val >= valid_range[0]) && (val <= valid_range[1])) ? 1 : 0;
                }

                if (this->verbose && (rank == 0))
                {
                    TECA_STATUS("Mask for array \""
                        << array_name << "\" will be generated using valid_range=["
                        << valid_range[0] << ", " << valid_range[1] << "]")
                }
            }
            else
            {
                if (this->verbose && (rank == 0))
                {
                    TECA_STATUS("Mask array for \"" << array_name
                        << "\" was requested but could not be computed. Attributes may"
                           " be missing a _FillValue, missing_value, valid_min, valid_max;"
                           " or valid_range. call enable_valid_range to enable the use of"
                           " valid_min, valid_max and valid_range attributes.")
                }
                continue;
            }

            // save the mask in the output
            std::string mask_name = array_name + "_valid";
            arrays->set(mask_name, mask);
            )

    }

    return out_mesh;
}
