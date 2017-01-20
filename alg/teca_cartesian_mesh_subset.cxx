#include "teca_cartesian_mesh_subset.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// --------------------------------------------------------------------------
teca_cartesian_mesh_subset::teca_cartesian_mesh_subset()
    : bounds({0.0,0.0,0.0,0.0,0.0,0.0}), cover_bounds(false)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_subset::~teca_cartesian_mesh_subset()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cartesian_mesh_subset::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_subset":prefix));

    opts.add_options()
        TECA_POPTS_GET(vector<double>, prefix, bounds,
            "bounding box given by x0,x1,y0,y1,z0,z1")
        TECA_POPTS_GET(bool, prefix, cover_bounds,
            "(T)use smallest subset covering or (F)largest "
            "subset contained by bounds")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_subset::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, vector<double>, prefix, bounds)
    TECA_POPTS_SET(opts, bool, prefix, cover_bounds)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_subset::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_subset::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata coords;
    const_p_teca_variant_array x;
    const_p_teca_variant_array y;
    const_p_teca_variant_array z;

    if (input_md[0].get("coordinates", coords)
        || !(x = coords.get("x")) || !(y = coords.get("y"))
        || !(z = coords.get("z")))
    {
        TECA_ERROR("Input metadata has invalid coordinates")
        return teca_metadata();
    }

    this->extent.resize(6, 0UL);
    if (teca_coordinate_util::bounds_to_extent(
        this->bounds.data(), x, y, z, this->extent.data()))
    {
        TECA_ERROR("Failed to convert bounds to extent")
        return teca_metadata();
    }

    teca_metadata out_md(input_md[0]);
    out_md.insert("whole_extent", this->extent);
    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cartesian_mesh_subset::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs(1, request);

    up_reqs[0].insert("extent", this->extent);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_subset::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_subset::execute" << endl;
#endif
    (void)port;
    (void)request;

    p_teca_cartesian_mesh in_target
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(input_data[0]));

    if (!in_target)
    {
        TECA_ERROR("invalid input dataset")
        return nullptr;
    }

    // pass input through via shallow copy
    p_teca_cartesian_mesh target = teca_cartesian_mesh::New();
    target->shallow_copy(in_target);

    return target;
}
