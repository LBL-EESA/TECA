#include "teca_shape_file_mask.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"
#include "teca_geometry.h"
#include "teca_shape_file_util.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;

//#define TECA_DEBUG

using poly_coord_t = double;

struct teca_shape_file_mask::internals_t
{
    std::vector<teca_geometry::polygon<poly_coord_t>> polys;
};


// --------------------------------------------------------------------------
teca_shape_file_mask::teca_shape_file_mask() : normalize_coordinates(0), internals(new teca_shape_file_mask::internals_t)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_shape_file_mask::~teca_shape_file_mask()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_shape_file_mask::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_shape_file_mask":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, shape_file,
            "Path and file name to one of the *.shp/*.shx files")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, mask_variables,
            "Set the names of the variables to store the generated mask in."
            " Each name is assigned a reference to the mask.")
        TECA_POPTS_GET(int, prefix, normalize_coordinates,
            "Set this flag to apply coordinate transform such that the x"
            " coordinates range from [0, 360] and the y coordinates range from"
            " [-90, 90]")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_shape_file_mask::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, shape_file)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, mask_variables)
    TECA_POPTS_SET(opts, int, prefix, normalize_coordinates)
}
#endif

// --------------------------------------------------------------------------
void teca_shape_file_mask::set_modified()
{
   this->teca_algorithm::set_modified();
   this->internals->polys.clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_shape_file_mask::get_mask_array_attributes(unsigned long size)
{
    unsigned int centering = teca_array_attributes::point_centering;

    // construct output attributes
    teca_array_attributes mask_atts(
        teca_variant_array_code<char>::get(),
        centering, size, {1,1,0,0}, "none", "", "Mask array generated from "
        + teca_file_util::filename(this->shape_file));

    return mask_atts;
}

// --------------------------------------------------------------------------
teca_metadata teca_shape_file_mask::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_shape_file_mask::get_output_metadata" << std::endl;
#endif
    (void)port;

    // validate runtime provided settings
    unsigned int n_mask_vars = this->mask_variables.size();
    if (n_mask_vars == 0)
    {
        TECA_FATAL_ERROR("The names of the mask_variables were not provided")
        return teca_metadata();
    }

    if (this->shape_file.empty())
    {
        TECA_FATAL_ERROR("A shape file was not provided")
        return teca_metadata();

    }

    // load the polygons
    if (this->internals->polys.empty() &&
        teca_shape_file_util::load_polygons(this->get_communicator(),
        this->shape_file, this->internals->polys, this->normalize_coordinates,
        this->verbose))
    {
        TECA_FATAL_ERROR("Failed to read polygons from \"" << this->shape_file << "\"")
        return teca_metadata();
    }

    // pass metadata from the input mesh through.
    const teca_metadata &mesh_md = input_md[0];
    teca_metadata out_md(mesh_md);

    // add the mask arrays we will generate
    for (unsigned int i = 0; i < n_mask_vars; ++i)
        out_md.append("variables", this->mask_variables[i]);

    // insert attributes to enable this to be written by the CF writer
    teca_metadata attributes;
    out_md.get("attributes", attributes);

    teca_metadata array_atts = this->get_mask_array_attributes(0);

    // add one for each output
    for (unsigned int i = 0; i < n_mask_vars; ++i)
        attributes.set(this->mask_variables[i], array_atts);

    // update the attributes collection
    out_md.set("attributes", attributes);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_shape_file_mask::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
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
    unsigned int n_mask_vars = this->mask_variables.size();
    for (unsigned int i = 0; i < n_mask_vars; ++i)
        arrays.erase(this->mask_variables[i]);

    req.set("arrays", arrays);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_shape_file_mask::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "teca_shape_file_mask::execute" << std::endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Failed to compute surface pressure. The dataset is"
            " not a teca_mesh")
        return nullptr;
    }

    // get the coordinate arrays.
    const_p_teca_variant_array x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array z = in_mesh->get_z_coordinates();

    if (z->size() > 1)
    {
        TECA_FATAL_ERROR("The shape file mask requires 2D data but 3D data was found")
        return nullptr;
    }

    // allocate the mask array
    unsigned long nx = x->size();
    unsigned long ny = y->size();
    unsigned long nxy = nx*ny;
    unsigned long nxm1 = nx - 1;
    unsigned long nym1 = ny - 1;

    poly_coord_t mesh_bounds[6] = {0};

    x->get(0,    mesh_bounds[0]);
    x->get(nxm1, mesh_bounds[1]);
    y->get(0,    mesh_bounds[2]);
    y->get(nym1, mesh_bounds[3]);

    auto [mask, p_mask] = ::New<teca_char_array>(nxy, char(0));

    // visit each polygon
    unsigned long np = this->internals->polys.size();
    for (unsigned long p = 0; p < np; ++p)
    {

        // get the polygon's axis aligned bounding box
        poly_coord_t poly_bounds[6] = {0};
        this->internals->polys[p].get_bounds(poly_bounds);

        // interset with the mesh bounds.
        poly_coord_t bounds[6] = {
            std::max(poly_bounds[0], mesh_bounds[0]),
            std::min(poly_bounds[1], mesh_bounds[1]),
            std::max(poly_bounds[2], mesh_bounds[2]),
            std::min(poly_bounds[3], mesh_bounds[3]),
            0.0, 0.0};

        // check for empty intersection. in that case nothing to do
        if ((bounds[0] > bounds[1]) || (bounds[2] > bounds[3]))
            continue;

        // convert the overlapping region to the mesh extents
        unsigned long extent[6] = {0};
        if (teca_coordinate_util::bounds_to_extent(bounds, x, y, z, extent))
        {
            TECA_FATAL_ERROR("Failed to convert polygon " << p << " bounds ["
                << bounds[0] << ", " << bounds[1] << ", " << bounds[2]
                << ", " << bounds[3] << " to a valid mesh extent")
            continue;
        }

        // test each point in the overlapping region for intersection with the
        // polygon
        VARIANT_ARRAY_DISPATCH_FP(x.get(),

            assert_type<CTT>(y);

            auto [sp_x, p_x, sp_y, p_y] = get_cpu_accessible<CTT>(x, y);

            for (unsigned long j = extent[2]; j <= extent[3]; ++j)
            {
                for (unsigned long i = extent[0]; i <= extent[1]; ++i)
                {
                    if (this->internals->polys[p].inside(p_x[i], p_y[j]))
                        p_mask[j*nx + i] = 1;
                }
            }
            )
    }

    // pass incoming data
    p_teca_cartesian_mesh out_mesh =
        std::static_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_cartesian_mesh>(in_mesh)->new_shallow_copy());

    // pass a reference to the mask
    unsigned int n_mask_vars = this->mask_variables.size();
    for (unsigned int i = 0; i < n_mask_vars; ++i)
        out_mesh->get_point_arrays()->set(this->mask_variables[i], mask);

    // add attributes
    teca_metadata attributes;
    out_mesh->get_attributes(attributes);

    teca_metadata array_atts = this->get_mask_array_attributes(nxy);

    // add one for each output
    for (unsigned int i = 0; i < n_mask_vars; ++i)
        attributes.set(this->mask_variables[i], array_atts);

    // update the attributes collection
    out_mesh->set_attributes(attributes);

    return out_mesh;
}
