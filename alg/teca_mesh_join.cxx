#include "teca_mesh_join.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_coordinate_util.h"

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
teca_mesh_join::teca_mesh_join()
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_mesh_join::~teca_mesh_join()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_mesh_join::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_mesh_join":prefix));

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_mesh_join::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);
}
#endif


// --------------------------------------------------------------------------
teca_metadata teca_mesh_join::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_mesh_join::get_output_metadata" << std::endl;
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

    teca_metadata target_coords;
    input_md[md_target].get("coordinates", target_coords);

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

        teca_metadata source_coords;
        input_md[md_src].get("coordinates", source_coords);

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

                teca_metadata coords;
                source_coords.get(src_var, coords);
                target_coords.set(src_var, coords);
            }
        }
    }

    // update with merged lists
    output_md.set("variables", target_vars);
    output_md.set("attributes", target_atts);
    output_md.set("coordinates", target_coords);

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_mesh_join::get_upstream_request(
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
const_p_teca_dataset teca_mesh_join::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_mesh_join::execute" << std::endl;
#endif
    (void)port;

    unsigned int n_in = this->get_number_of_input_connections();

    p_teca_cartesian_mesh in_target
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(input_data[0]));

    if (!in_target)
    {
        TECA_FATAL_ERROR("invalid input. target invalid")
        return nullptr;
    }

    // create the output
    p_teca_cartesian_mesh target = teca_cartesian_mesh::New();
    target->shallow_copy(in_target);

    // get the coordinate and data arrays
    const_p_teca_variant_array target_xc = target->get_x_coordinates();
    const_p_teca_variant_array target_yc = target->get_y_coordinates();
    const_p_teca_variant_array target_zc = target->get_z_coordinates();
    p_teca_array_collection target_ac = target->get_point_arrays();

    // get the attributes
    teca_metadata target_atts;
    target->get_attributes(target_atts);

    unsigned long target_nx = target_xc->size();
    unsigned long target_ny = target_yc->size();
    unsigned long target_nz = target_zc->size();

    unsigned long target_ihi = target_nx - 1;
    unsigned long target_jhi = target_ny - 1;

    double tx0, tx1;
    target_xc->get(0, tx0);
    target_xc->get(target_ihi, tx1);

    double ty0, ty1;
    target_yc->get(0, ty0);
    target_yc->get(target_jhi, ty1);

    // get the list of arrays to move
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    auto it = req_arrays.begin();
    auto end = req_arrays.end();
    for (; it != end; ++it)
    {
        if (!target->get_point_arrays()->has(*it))
        {
            bool array_found = false;
            for (unsigned int i = 1; i < n_in; ++i)
            {
                const_p_teca_cartesian_mesh source
                    = std::dynamic_pointer_cast<const teca_cartesian_mesh>
                        (input_data[i]);

                if (!source)
                {
                    TECA_FATAL_ERROR("invalid input. source invalid")
                    return nullptr;
                }

                if (source->get_point_arrays()->has(*it))
                {
                    const_p_teca_variant_array source_xc = source->get_x_coordinates();
                    const_p_teca_variant_array source_yc = source->get_y_coordinates();
                    const_p_teca_variant_array source_zc = source->get_z_coordinates();
                    const_p_teca_array_collection source_ac = source->get_point_arrays();

                    unsigned long source_shape[4] = {0};
                    source->get_array_shape(*it, source_shape);

                    unsigned long source_nx = source_shape[0];
                    unsigned long source_ny = source_shape[1];
                    unsigned long source_nz = source_shape[2];
                    unsigned long source_ihi = source_nx - 1;
                    unsigned long source_jhi = source_ny - 1;

                    double sx0, sx1;
                    source_xc->get(0, sx0);
                    source_xc->get(source_ihi, sx1);

                    double sy0, sy1;
                    source_yc->get(0, sy0);
                    source_yc->get(source_jhi, sy1);

                    // copy when the input and output meshes are the same. the meshes are the
                    // same when they span the same world space and have the same resolution
                    if ((source_nx == target_nx) && (source_ny == target_ny) && (source_nz == target_nz) &&
                        teca_coordinate_util::equal(sx0, tx0) && teca_coordinate_util::equal(sx1, tx1) &&
                        teca_coordinate_util::equal(sy0, ty0) && teca_coordinate_util::equal(sy1, ty1))
                    {
                        // move the array attributes
                        teca_metadata source_atts;
                        source->get_attributes(source_atts);

                        teca_metadata array_atts;
                        if (!source_atts.get(*it, array_atts))
                            target_atts.set(*it, array_atts);

                        if (this->verbose)
                        {
                            TECA_STATUS("\""<< *it << "\". Identical dimensions[x,y,z]: "
                                << source_nx << "/" << target_nx << ", "
                                << source_ny << "/" << target_ny << ", "
                                << source_nz << "/" << target_nz
                                << " and identical bounds[x,y]: "
                                << sx0 << "/" << tx0 << ", " << sx1 << "/" << tx1 << ", "
                                << sy0 << "/" << ty0 << ", " << sy1 << "/" << ty1
                                << ". Copying data.")
                        }

                        const_p_teca_variant_array source_a = source_ac->get(*it);
                        p_teca_variant_array target_a = source_a->new_copy();
                        target_ac->set(*it, target_a);

                        array_found = true;
                        break;
                    }
                    else
                    {
                        TECA_FATAL_ERROR("\""<< *it << "\". Not identical dimensions[x,y,z]: "
                                << source_nx << "/" << target_nx << ", "
                                << source_ny << "/" << target_ny << ", "
                                << source_nz << "/" << target_nz
                                << " or not identical bounds[x,y]: "
                                << sx0 << "/" << tx0 << ", " << sx1 << "/" << tx1 << ", "
                                << sy0 << "/" << ty0 << ", " << sy1 << "/" << ty1)
                        return nullptr;
                    }
                }
            }

            if (!array_found)
            {
                TECA_FATAL_ERROR("Array \"" << *it
                    << "\" is not present on any of the inputs")
                return nullptr;
            }
        }
    }

    target->set_attributes(target_atts);

    return target;
}
