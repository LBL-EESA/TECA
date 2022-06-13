#include "teca_vertical_reduction.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

// --------------------------------------------------------------------------
teca_vertical_reduction::teca_vertical_reduction()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_vertical_reduction::~teca_vertical_reduction()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vertical_reduction::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_vertical_reduction":prefix));

    opts.add_options()
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, dependent_variables,
            "list of arrays needed to compute the derived quantity")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, derived_variables,
            "name of the derived quantity")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vertical_reduction::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, dependent_variables)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, derived_variables)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_vertical_reduction::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_vertical_reduction::get_output_metadata" << std::endl;
#endif
    (void)port;

    if (this->derived_variables.empty())
    {
        TECA_FATAL_ERROR("A derived variable was not specififed")
        return teca_metadata();
    }

    // add in the arrays we will generate and their attributes
    teca_metadata out_md(input_md[0]);

    teca_metadata attributes;
    out_md.get("attributes", attributes);

    size_t n_derived = this->derived_variables.size();
    for (size_t i = 0; i < n_derived; ++i)
    {
        out_md.append("variables", this->derived_variables[i]);

        attributes.set(this->derived_variables[i],
            (teca_metadata)this->derived_variable_attributes[i]);
    }
    out_md.set("attributes", attributes);

    // get the input extents
    unsigned long whole_extent[6] = {0};
    if (out_md.get("whole_extent", whole_extent, 6))
    {
        TECA_FATAL_ERROR("Metadata is missing whole_whole_extent")
        return teca_metadata();
    }

    // set the output extent, with vertical dim reduced
    whole_extent[4] = whole_extent[5] = 0;
    out_md.set("whole_extent", whole_extent, 6);

    // fix bounds if it is present
    double bounds[6] = {0.0};
    if (out_md.get("bounds", bounds, 6) == 0)
    {
        bounds[4] = bounds[5] = 0.0;
        out_md.set("bounds", bounds, 6);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_vertical_reduction::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;

    std::vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    // transform extent, add back the vertical dimension
    const teca_metadata md = input_md[0];

    // get the whole extent and bounds
    double bounds[6] = {0.0};
    unsigned long whole_extent[6] = {0};
    if (teca_coordinate_util::get_cartesian_mesh_extent(md,
        whole_extent, bounds))
    {
        TECA_FATAL_ERROR("Failed to get input data set extent")
        return up_reqs;
    }

    /*bool has_bounds = request.has("bounds");
    bool has_extent = request.has("extent");*/

    // restore vertical bounds
    double bounds_up[6] = {0.0};
    unsigned long extent_up[6] = {0};
    if (request.get("bounds", bounds_up, 6) == 0)
    {
        bounds_up[4] = bounds[4];
        bounds_up[5] = bounds[5];
        req.set("bounds", bounds_up, 6);
    }

    // restore vertical extent
    else if (request.get("extent", extent_up, 6) == 0)
    {
        extent_up[4] = whole_extent[4];
        extent_up[5] = whole_extent[5];
        req.set("extent", extent_up, 6);
    }
    // no subset requested, request all the data
    else
    {
        req.set("extent", whole_extent);
    }

    // get the list of variable available. we need to see if
    // the valid value mask is available and if so request it
    std::set<std::string> variables;
    if (md.get("variables", variables))
    {
        TECA_FATAL_ERROR("Metadata issue. variables is missing")
        return up_reqs;
    }

    // add the dependent variables into the requested arrays
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    int n_dep_vars = this->dependent_variables.size();
    for (int i = 0; i < n_dep_vars; ++i)
    {
        const std::string &dep_var = this->dependent_variables[i];

        // request the array needed for the calculation
        arrays.insert(dep_var);

        // request the valid value mask if they are available.
        std::string mask_var = dep_var + "_valid";
        if (variables.count(mask_var))
            arrays.insert(mask_var);
    }

    // capture the arrays we produce
    size_t n_derived = this->derived_variables.size();
    for (size_t i = 0; i < n_derived; ++i)
       arrays.erase(this->derived_variables[i]);

    // update the request
    req.set("arrays", arrays);

    // send it up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vertical_reduction::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_vertical_reduction::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("teca_mesh is required")
        return nullptr;
    }

    // construct the output
    p_teca_mesh out_mesh =
        std::dynamic_pointer_cast<teca_mesh>(in_mesh->new_instance());

    // copy metadata
    out_mesh->copy_metadata(in_mesh);

    // copy the coordinates
    // out_mesh->copy_coordinates(in_mesh);

    // fix the metadata
    teca_metadata out_md = out_mesh->get_metadata();

    // fix whole extent
    unsigned long whole_extent[6] = {0};
    if (out_md.get("whole_extent", whole_extent, 6) == 0)
    {
        whole_extent[4] = whole_extent[5] = 0;
        out_md.set("whole_extent", whole_extent, 6);
    }

    // fix extent
    unsigned long extent[6] = {0};
    if (out_md.get("extent", extent, 6) == 0)
    {
        extent[4] = extent[5] = 0;
        out_md.set("extent", extent, 6);
    }

    // fix bounds
    double bounds[6] = {0};
    if (out_md.get("bounds", bounds, 6) == 0)
    {
        bounds[4] = bounds[5] = 0.0;
        out_md.set("bounds", bounds, 6);
    }

    out_mesh->set_metadata(out_md);


    // fix the z axis
    p_teca_cartesian_mesh cart_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(out_mesh);

    if (cart_mesh)
    {
        std::string z_var;
        cart_mesh->get_z_coordinate_variable(z_var);

        const_p_teca_variant_array in_z = cart_mesh->get_z_coordinates();

        p_teca_variant_array out_z = in_z->new_instance(1);
        cart_mesh->set_z_coordinates(z_var, out_z);
    }

    return out_mesh;
}
