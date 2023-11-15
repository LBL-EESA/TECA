#include "teca_vertical_coordinate_transform.h"

#include "teca_arakawa_c_grid.h"
#include "teca_curvilinear_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::cos;

using namespace teca_variant_array_util;

//#define TECA_DEBUG

namespace {

/*  unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxy = nx*ny;*/

// the inputs xi, yi are 2D fields, eta is a 1D field
// the outputs xo,yo, and phd are 3D fields
template <typename num_t>
void transform_wrf_v3(unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, const num_t *xi,
    const num_t *yi, const num_t *eta, const num_t *ps, num_t pt,
    num_t* xo, num_t *yo, num_t *ph)
{
    unsigned long nxy_bytes = nxy*sizeof(num_t);
    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long knxy = k*nxy;

        // copy x
        memcpy(xo + knxy, xi, nxy_bytes);

        // copy y
        memcpy(yo + knxy, yi, nxy_bytes);

        // transform z
        num_t eta_k = eta[k];
        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jnx = j*nx;
            unsigned long knxy_jnx = knxy + jnx;

            // transform from mass vertical coordinate into hydrostatic dry pressure
            // see "A descritpion  of Advanced Research WRF Model Version 4" page 8
            // note: this is the WRF ARW 3 coordinate system
            const num_t *ps_jnx = ps + jnx;
            num_t *ph_knxy_jnx = ph + knxy_jnx;
            for (unsigned long i = 0; i < nx; ++i)
                ph_knxy_jnx[i] = eta_k*(ps_jnx[i] - pt) + pt;
        }
    }
}

};


// --------------------------------------------------------------------------
teca_vertical_coordinate_transform::teca_vertical_coordinate_transform() :
    mode(mode_wrf_v3)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_vertical_coordinate_transform::~teca_vertical_coordinate_transform()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vertical_coordinate_transform::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_vertical_coordinate_transform":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, mode,
            "Sets the coordinate transform mode. The modes are: mode_wrf_v3")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vertical_coordinate_transform::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, mode)
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_vertical_coordinate_transform::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_vertical_coordinate_transform::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata out_md(input_md[0]);

    // get coordinate metadata
    teca_metadata coords;
    if (out_md.get("coordinates", coords))
    {
        TECA_FATAL_ERROR("metadata issue, missing coordinate metadata")
        return teca_metadata();
    }

    // get array metadata
    teca_metadata atrs;
    if (out_md.get("attributes", atrs))
    {
        TECA_FATAL_ERROR("failed to get array attributes")
        return teca_metadata();
    }

    switch (this->mode)
    {
        case mode_wrf_v3:
        {
            // update the z coordinate axes variable names so that down
            // stream algorithms correctly identify them
            coords.set("m_z_variable", "ZPDM");
            coords.set("w_z_varibale", "ZPDW");

            // pass metadata for the atrrays we generate
            teca_metadata ps_atts;
            if (atrs.get("PSFC", ps_atts))
            {
                TECA_FATAL_ERROR("failed to get PSFC attributes")
                return teca_metadata();
            }
            atrs.set("ZPDM", ps_atts);
            atrs.set("ZPDW", ps_atts);
            break;
        }
        default:
        {
            TECA_FATAL_ERROR("Invlaid mode " << this->mode)
            return teca_metadata();
        }
    }

    out_md.set("coordinates", coords);
    out_md.set("attributes", atrs);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_vertical_coordinate_transform::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    vector<teca_metadata> up_reqs;

    // copy the incoming request to preserve the downstream
    // requirements and add the arrays we need
    teca_metadata req(request);

    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);

    switch (this->mode)
    {
        case mode_wrf_v3:
        {
            arrays.insert("PSFC");
            arrays.insert("P_TOP");
            break;
        }
        default:
            TECA_FATAL_ERROR("Invlaid mode " << this->mode)
            return up_reqs;
    }

    // update the request
    req.set("arrays", arrays);

    // if/when bounds based requests are
    // implemented the transform from pressure coordinates to
    // sigma coordinates would be handled here.

    // send it up
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vertical_coordinate_transform::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_vertical_coordinate_transform::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_arakawa_c_grid in_mesh
        = std::dynamic_pointer_cast<const teca_arakawa_c_grid>(input_data[0]);

    if (!in_mesh)
    {
        TECA_FATAL_ERROR("teca_arakawa_c_grid is required")
        return nullptr;
    }

    // create the output mesh. it is a curvilinear mesh because
    // of the fully 3D realization of the coordinate arrays.
    // also cell centered values are moved to node centering, i.e.
    // this is the dual mesh.
    p_teca_curvilinear_mesh out_mesh = teca_curvilinear_mesh::New();
    out_mesh->copy_metadata(in_mesh);

    switch (this->mode)
    {
        case mode_wrf_v3:
        {
            // get the input coordinate system
            std::string x_coord_name;
            std::string y_coord_name;

            in_mesh->get_m_x_coordinate_variable(x_coord_name);
            in_mesh->get_m_y_coordinate_variable(y_coord_name);

            const_p_teca_variant_array xi = in_mesh->get_m_x_coordinates();
            const_p_teca_variant_array yi = in_mesh->get_m_y_coordinates();
            const_p_teca_variant_array eta = in_mesh->get_m_z_coordinates();

            const_p_teca_variant_array pt = in_mesh->get_information_arrays()->get("P_TOP");
            if (!pt)
            {
                TECA_FATAL_ERROR("Failed to get P_TOP")
                return nullptr;
            }

            const_p_teca_variant_array ps = in_mesh->get_cell_arrays()->get("PSFC");
            if (!ps)
            {
                TECA_FATAL_ERROR("Failed to get PSFC")
                return nullptr;
            }

            // get the mesh dimensions
            unsigned long extent[6] = {0};
            in_mesh->get_extent(extent);

            unsigned long nx = extent[1] - extent[0] + 1;
            unsigned long ny = extent[3] - extent[2] + 1;
            unsigned long nz = extent[5] - extent[4] + 1;
            unsigned long nxy = nx*ny;
            unsigned long nxyz = nxy*nz;

            // allocate the output coordinates
            p_teca_variant_array xo;
            p_teca_variant_array yo;
            p_teca_variant_array ph;

            VARIANT_ARRAY_DISPATCH(xi.get(),

                auto [tmp_xo, pxo] = ::New<TT>(nxyz);
                auto [tmp_yo, pyo] = ::New<TT>(nxyz);
                auto [tmp_ph, pph] = ::New<TT>(nxyz);

                assert_type<CTT>(pt, ps, xi, yi, eta);
                auto [sppt, ppt] = get_host_accessible<CTT>(pt);
                auto [spps, pps] = get_host_accessible<CTT>(ps);
                auto [spxi, pxi] = get_host_accessible<CTT>(xi);
                auto [spyi, pyi] = get_host_accessible<CTT>(yi);
                auto [speta, peta] = get_host_accessible<CTT>(eta);

                sync_host_access_any(pt, ps, xi, yi, eta);

                ::transform_wrf_v3(nx, ny, nz, nxy, pxi, pyi,
                    peta, pps, ppt[0], pxo, pyo, pph);

                xo = tmp_xo;
                yo = tmp_yo;
                ph = tmp_ph;
                )

            // pass coordinates to output
            out_mesh->set_x_coordinates(x_coord_name, xo);
            out_mesh->set_y_coordinates(y_coord_name, yo);
            out_mesh->set_z_coordinates("ZPHM", ph);

            // pass arrays to the output
            out_mesh->get_point_arrays() =
                std::const_pointer_cast<teca_array_collection>
                    (in_mesh->get_cell_arrays());

            break;
        }
        default:
        {
            TECA_FATAL_ERROR("Invlaid mode " << this->mode)
            return nullptr;
        }
    }

    return out_mesh;
}
