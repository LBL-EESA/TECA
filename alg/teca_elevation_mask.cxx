#include "teca_elevation_mask.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include "teca_dataset_source.h"
#include "teca_dataset_capture.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_index_executive.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG

using namespace teca_variant_array_util;


struct teca_elevation_mask::internals_t
{
    // compute the valid value mask such that for each point the mask
    // is 1 where the mesh point is above the surface of the Earth and
    // 0 otherwise
    template<typename mask_t, typename elev_num_t, typename mesh_num_t>
    static void mask_by_surface_elevation(
        size_t nx, size_t ny, size_t nz,
        mask_t * __restrict__ mask,
        const elev_num_t * __restrict__ surface_elev,
        const mesh_num_t * __restrict__ mesh_height)
    {
        size_t nxy = nx*ny;
        for (size_t k = 0; k < nz; ++k)
        {
            const mesh_num_t * __restrict__ mesh_height_k = mesh_height + k*nxy;
            mask_t * __restrict__ mask_k = mask + k*nxy;
            for (size_t q = 0; q < nxy; ++q)
            {
                mask_k[q] = mesh_height_k[q] >= (mesh_num_t)surface_elev[q] ? mask_t(1) : mask_t(0);
            }
        }
    }
};


// --------------------------------------------------------------------------
teca_elevation_mask::teca_elevation_mask() :
    mesh_height_variable("zg"), surface_elevation_variable("z")
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_elevation_mask::~teca_elevation_mask()
{
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_elevation_mask::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_elevation_mask":prefix));

    opts.add_options()

        TECA_POPTS_GET(std::string, prefix, surface_elevation_variable,
            "Set the name of the variable containing surface elevation"
            " values in meters above mean sea level")

        TECA_POPTS_GET(std::string, prefix, mesh_height_variable,
            "Set the name of the variable containing point wise mesh height"
            " values in meters above mean sea level")

        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, mask_variables,
            "Set the names of the variables to store the generated mask in."
            " Each name is assigned a reference to the mask.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_elevation_mask::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, surface_elevation_variable)
    TECA_POPTS_SET(opts, std::string, prefix, mesh_height_variable)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, mask_variables)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_elevation_mask::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_elevation_mask::get_output_metadata" << endl;
#endif
    (void)port;

    // validate runtime provided settings
    unsigned int n_mask_vars = this->mask_variables.size();
    if (n_mask_vars == 0)
    {
        TECA_FATAL_ERROR("The names of the mask_variables were not provided")
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

    teca_metadata mesh_height_atts;
    if (attributes.get(this->mesh_height_variable, mesh_height_atts))
    {
        TECA_WARNING("Failed to get mesh_height_variable \""
            << this->mesh_height_variable << "\" attrbibutes."
            " Writing the result will not be possible")
    }
    else
    {
        // get the centering and size from the array
        unsigned int centering = 0;
        mesh_height_atts.get("centering", centering);

        unsigned long size = 0;
        mesh_height_atts.get("size", size);

        auto dim_active = teca_array_attributes::xyzt_active();
        mesh_height_atts.get("mesh_dim_active", dim_active);

        // construct output attributes
        teca_array_attributes mask_atts(
            teca_variant_array_code<char>::get(), centering, size,
            dim_active, "none", "", "elevation mask");

        // add one for each output
        for (unsigned int i = 0; i < n_mask_vars; ++i)
            attributes.set(this->mask_variables[i], (teca_metadata)mask_atts);

        // update the attributes collection
        out_md.set("attributes", attributes);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_elevation_mask::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // get the names of the arrays we need to request
    if (this->mesh_height_variable.empty())
    {
        TECA_FATAL_ERROR("The mesh_height_variable was not specified")
        return up_reqs;
    }

    if (this->surface_elevation_variable.empty())
    {
        TECA_FATAL_ERROR("The surface_elevation_variable was not specified")
        return up_reqs;
    }

    // need to make the request for the surface elevation field using bounds
    double req_bounds[6] = {0.0};
    if (request.get("bounds", req_bounds, 6))
    {
        // bounds not specified, try to get an extent and convert to a bounds
        unsigned long req_extent[6];
        if (request.get("extent", req_extent, 6))
        {
            TECA_FATAL_ERROR("Neither bounds nor extent were specified in the request")
            return up_reqs;
        }

        const teca_metadata &md = input_md[0];

        teca_metadata coords;
        p_teca_variant_array x,y;

        if (md.get("coordinates", coords) ||
            !(x = coords.get("x")) || !(y = coords.get("y")))
        {
            TECA_FATAL_ERROR("Failed to get mesh coordinates")
            return up_reqs;
        }

        x->get(req_extent[0], req_bounds[0]);
        x->get(req_extent[1], req_bounds[1]);
        y->get(req_extent[2], req_bounds[2]);
        y->get(req_extent[3], req_bounds[3]);
    }

    // input port 0 will source the mesh height field, and any other data
    // requested by the down stream.  copy the incoming request to preserve the
    // downstream requirements and add the mesh height variable
    teca_metadata req_0(request);

    std::set<std::string> mesh_arrays;
    if (req_0.has("arrays"))
        req_0.get("arrays", mesh_arrays);

    mesh_arrays.insert(this->mesh_height_variable);

    // intercept request for our output
    int n_mask_vars = this->mask_variables.size();
    for (int i = 0; i < n_mask_vars; ++i)
        mesh_arrays.erase(this->mask_variables[i]);

    req_0.set("arrays", mesh_arrays);


    // input port 1 provides the surface elevation field, request it
    // preserve bounds etc
    const teca_metadata &elev_md = input_md[1];

    std::string req_key;
    if (elev_md.get("index_request_key", req_key))
    {
        TECA_FATAL_ERROR("Metadata is missing \"index_request_key\"")
        return up_reqs;
    }

    // surface elevations don't change over the timescale of concern
    // always request index 0
    teca_metadata req_1;
    req_1.set(req_key, {0ul, 0ul});
    req_1.set("index_request_key", req_key);

    // request the surface elevation
    std::vector<std::string> elev_arrays(1, this->surface_elevation_variable);
    req_1.set("arrays", elev_arrays);

    // at the bounds of interest
    req_1.set("bounds", req_bounds, 6);

    // package the requests and send them up
    up_reqs.push_back(req_0);
    up_reqs.push_back(req_1);


    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_elevation_mask::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_elevation_mask::execute" << endl;
#endif
    (void)port;
    (void)request;

    // check for an error upstream
    if ((input_data.size() != 2) || !input_data[0] || !input_data[1])
    {
        TECA_FATAL_ERROR("Invalid inputs detected")
        return nullptr;
    }

    // get the input 3D mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("Data to mask on input port 0 is not a"
            " teca_cartesian_mesh. Got " << input_data[0]->get_class_name())
        return nullptr;
    }

    // get the mesh dimensions
    unsigned long extent[6];
    in_mesh->get_extent(extent);

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    // get the mesh height, this is a 3d field with the altitude for
    // each mesh point
    const_p_teca_variant_array mesh_height =
        in_mesh->get_point_arrays()->get(this->mesh_height_variable);

    if (!mesh_height)
    {
        TECA_FATAL_ERROR("Mesh to mask is missing the height field \""
            << this->mesh_height_variable << "\"")
        return nullptr;
    }

    // get the surface elevations
    const_p_teca_cartesian_mesh in_elev
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[1]);

    if (!in_elev)
    {
        TECA_FATAL_ERROR("Data to mask on input port 0 is not a"
            " teca_cartesian_mesh. Got " << input_data[0]->get_class_name())
        return nullptr;
    }

    // get the surface elevation, this is a 2d field with surface altitude
    //
    // at each mesh point. regridding has been performed so that the horizontal
    // coordinates are the same as the 3d mesh for which masks will be generated
    const_p_teca_variant_array surface_elev =
        in_elev->get_point_arrays()->get(this->surface_elevation_variable);

    if (!surface_elev)
    {
        TECA_FATAL_ERROR("Surface elevation data has no array \""
            << this->surface_elevation_variable << "\"")
        return nullptr;
    }

    // compute the mask
    p_teca_char_array mask = teca_char_array::New(mesh_height->size());
    char *p_mask = mask->data();

    NESTED_VARIANT_ARRAY_DISPATCH(
        surface_elev.get(), _SURF,

        auto [sp_se, p_surface_elev] = get_host_accessible<CTT_SURF>(surface_elev);

        NESTED_VARIANT_ARRAY_DISPATCH(
            mesh_height.get(), _MESH,

            auto [sp_mh, p_mesh_height] = get_host_accessible<CTT_MESH>(mesh_height);

            sync_host_access_any(surface_elev, mesh_height);

            internals_t::mask_by_surface_elevation(nx, ny, nz,
                p_mask, p_surface_elev, p_mesh_height);

            )
        )

    // allocate the output mesh
    p_teca_cartesian_mesh out_mesh = std::dynamic_pointer_cast<teca_cartesian_mesh>
        (std::const_pointer_cast<teca_cartesian_mesh>(in_mesh)->new_shallow_copy());

    // store the results under the requested names
    int n_mask_vars = this->mask_variables.size();
    for (int i = 0; i < n_mask_vars; ++i)
    {
       out_mesh->get_point_arrays()->set(this->mask_variables[i], mask);
    }

    return out_mesh;
}
