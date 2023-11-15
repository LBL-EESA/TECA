#include "teca_face_to_cell_centering.h"

#include "teca_arakawa_c_grid.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

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

template <typename num_t>
void x_face_to_cell(unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy,  const num_t *fc, num_t *cc)
{
    unsigned long nxf = nx + 1;
    unsigned long nxyf = nxf*ny;

    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long knxy = k*nxy;
        unsigned long knxyf = k*nxyf;

        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jnx = j*nx;
            unsigned long jnxf = j*nxf;

            const num_t *src = fc + knxyf + jnxf;
            num_t *dest = cc + knxy + jnx;

            for (unsigned long i = 0; i < nx; ++i)
            {
                dest[i] = 0.5*(src[i] + src[i+1]);
            }
        }
    }
}

template <typename num_t>
void y_face_to_cell(unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy,  const num_t *fc, num_t *cc)
{
    unsigned long nyf = ny + 1;
    unsigned long nxyf = nx*nyf;

    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long knxy = k*nxy;
        unsigned long knxyf = k*nxyf;

        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jnx = j*nx;
            unsigned long j1nx = (j + 1)*nx;

            const num_t *src = fc + knxyf + jnx;
            const num_t *src1 = fc + knxyf + j1nx;

            num_t *dest = cc + knxy + jnx;

            for (unsigned long i = 0; i < nx; ++i)
            {
                dest[i] = 0.5*(src[i] + src1[i]);
            }
        }
    }
}

template <typename num_t>
void z_face_to_cell(unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy,  const num_t *fc, num_t *cc)
{
    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long knxy = k*nxy;
        unsigned long k1nxy = (k + 1)*nxy;

        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jnx = j*nx;

            const num_t *src = fc + knxy + jnx;
            const num_t *src1 = fc + k1nxy + jnx;

            num_t *dest = cc + knxy + jnx;

            for (unsigned long i = 0; i < nx; ++i)
            {
                dest[i] = 0.5*(src[i] + src1[i]);
            }
        }
    }
}

};


// --------------------------------------------------------------------------
teca_face_to_cell_centering::teca_face_to_cell_centering()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_face_to_cell_centering::~teca_face_to_cell_centering()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_face_to_cell_centering::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_face_to_cell_centering":prefix));

    /*opts.add_options()
        TECA_POPTS_GET(int, prefix, mode,
            "Set the coordinate transform mode. The valid modes"
            " are: mode_wrf_v3)")
        ;*/

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_face_to_cell_centering::set_properties(
    const string &prefix, variables_map &opts)
{
    //TECA_POPTS_SET(opts, int, prefix, mode)
    this->teca_algorithm::set_properties(prefix, opts);
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_face_to_cell_centering::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_face_to_cell_centering::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata out_md(input_md[0]);

    // get array attributes
    teca_metadata atrs;
    if (out_md.get("attributes", atrs))
    {
        TECA_FATAL_ERROR("failed to get array attributes")
        return teca_metadata();
    }

    // get the list of array names
    std::vector<std::string> arrays;
    if (out_md.get("variables", arrays))
    {
        TECA_FATAL_ERROR("failed to get array names")
        return teca_metadata();
    }

    size_t n_arrays = arrays.size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // get the name of the ith array
        const std::string &array_name = arrays[i];

        // get the array's attributes
        teca_metadata array_atrs;
        if (atrs.get(array_name, array_atrs))
        {
            TECA_FATAL_ERROR("failed to get the attributes for array "
                << i << " \"" << array_name << "\"")
            return teca_metadata();
        }

        // get the array's centering
        int centering = teca_array_attributes::invalid_value;
        if (array_atrs.get("centering", centering))
        {
            TECA_FATAL_ERROR("failed to get the centering for array "
                << i << " \"" << array_name << "\"")
            return teca_metadata();
        }

        // if this is a face centered array change to cell centering
        if ((centering == teca_array_attributes::x_face_centering)
            || (centering == teca_array_attributes::y_face_centering)
            || (centering == teca_array_attributes::z_face_centering))
        {
            array_atrs.set("centering",
                int(teca_array_attributes::cell_centering));
        }

        // update the array's attributes
        atrs.set(array_name, array_atrs);
    }

    // update the array attributes collection
    out_md.set("attributes", atrs);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_face_to_cell_centering::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;
    std::vector<teca_metadata> up_reqs;
    up_reqs.push_back(request);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_face_to_cell_centering::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_face_to_cell_centering::execute" << endl;
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

    // get the mesh dimensions
    unsigned long extent[6] = {0};
    in_mesh->get_extent(extent);

    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;
    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    // allocate the output mesh
    p_teca_arakawa_c_grid out_mesh = teca_arakawa_c_grid::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_arakawa_c_grid>(in_mesh));

    // convert x-face centering to cell centering
    p_teca_array_collection &x_face_arrays = out_mesh->get_x_face_arrays();
    p_teca_array_collection &cell_arrays = out_mesh->get_cell_arrays();
    size_t n_arrays = x_face_arrays->size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        std::string &array_name = x_face_arrays->get_name(i);
        p_teca_variant_array fc = x_face_arrays->get(i);
        p_teca_variant_array cc = fc->new_instance(nxyz);
        VARIANT_ARRAY_DISPATCH(fc.get(),

            auto [spfc, pfc] = get_host_accessible<CTT>(fc);
            auto [pcc] = data<TT>(cc);

            sync_host_access_any(fc);

            ::x_face_to_cell(nx, ny, nz, nxy, pfc, pcc);
            )

        x_face_arrays->remove(i);
        cell_arrays->append(array_name, cc);
    }

    // convert y-face centering to cell centering
    p_teca_array_collection &y_face_arrays = out_mesh->get_y_face_arrays();
    n_arrays = y_face_arrays->size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        std::string &array_name = y_face_arrays->get_name(i);
        p_teca_variant_array fc = y_face_arrays->get(i);
        p_teca_variant_array cc = fc->new_instance(nxyz);
        VARIANT_ARRAY_DISPATCH(fc.get(),

            auto [spfc, pfc] = get_host_accessible<CTT>(fc);
            auto [pcc] = data<TT>(cc);

            sync_host_access_any(fc);

            ::y_face_to_cell(nx, ny, nz, nxy, pfc, pcc);
            )

        y_face_arrays->remove(i);
        cell_arrays->append(array_name, cc);
    }

    // convert z-face centering to cell centering
    p_teca_array_collection &z_face_arrays = out_mesh->get_z_face_arrays();
    n_arrays = z_face_arrays->size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        std::string &array_name = z_face_arrays->get_name(i);
        p_teca_variant_array fc = z_face_arrays->get(i);
        p_teca_variant_array cc = fc->new_instance(nxyz);
        VARIANT_ARRAY_DISPATCH(fc.get(),

            auto [spfc, pfc] = get_host_accessible<CTT>(fc);
            auto [pcc] = data<TT>(cc);

            sync_host_access_any(fc);

            ::z_face_to_cell(nx, ny, nz, nxy, pfc, pcc);
            )

        z_face_arrays->remove(i);
        cell_arrays->append(array_name, cc);
    }

    return out_mesh;
}
