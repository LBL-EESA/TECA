#include "teca_normalize_coordinates.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

//#define TECA_DEBUG

struct teca_normalize_coordinates::internals_t
{
    internals_t() {}
    ~internals_t() {}

    static
    p_teca_variant_array normalize_axis(const const_p_teca_variant_array &x);

    static
    void normalize_extent(p_teca_variant_array out_x,
        p_teca_variant_array out_y, p_teca_variant_array out_z,
        unsigned long *whole_extent, unsigned long *extent_in,
        unsigned long *extent_out);

    static
    void normalize_variables(bool normalize_x,
        bool normalize_y, bool normalize_z, unsigned long *extent,
        p_teca_array_collection data);
};


// --------------------------------------------------------------------------
p_teca_variant_array teca_normalize_coordinates::internals_t::normalize_axis(
    const const_p_teca_variant_array &x)
{
    unsigned long nx = x->size();
    unsigned long x1 = nx - 1;

    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        x.get(), _C,

        const NT_C *px = dynamic_cast<TT_C*>(x.get())->get();
        if (px[x1] < px[0])
        {
            p_teca_variant_array xo = x->new_instance(nx);
            NT_C *pxo = static_cast<teca_variant_array_impl<NT_C>*>(xo.get())->get();

            pxo += x1;
            for (unsigned long i = 0; i < nx; ++i)
                pxo[-i] = px[i];

            return xo;
        }
        )
    return nullptr;
}

// --------------------------------------------------------------------------
void teca_normalize_coordinates::internals_t::normalize_extent(
    p_teca_variant_array out_x, p_teca_variant_array out_y,
    p_teca_variant_array out_z, unsigned long *whole_extent,
    unsigned long *extent_in, unsigned long *extent_out)
{
#if defined(TECA_DEBUG)
    cerr << "out=[" << out_x << ", " << out_y << ", " << out_z << "]" << endl
        << "whole_extent=[" << whole_extent[0] << ", " << whole_extent[1] << ", "
        << whole_extent[2] << ", " << whole_extent[3] << ", " << whole_extent[4]
        << ", " << whole_extent[5] << "]" << endl << "extent_in=[" << extent_in[0]
        << ", " << extent_in[1] << ", " << extent_in[2] << ", " << extent_in[3]
        << ", " << extent_in[4] << ", " << extent_in[5] << "]" << endl;
#endif

    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    // transform the extent from outd coordinates back to
    // original
    if (out_x)
    {
        unsigned long wnx = whole_extent[1] - whole_extent[0];
        extent_out[0] = wnx - extent_in[1];
        extent_out[1] = wnx - extent_in[0];
    }

    if (out_y)
    {
        unsigned long wny = whole_extent[3] - whole_extent[2];
        extent_out[2] = wny - extent_in[3];
        extent_out[3] = wny - extent_in[2];
    }

    if (out_z)
    {
        unsigned long wnz = whole_extent[5] - whole_extent[4];
        extent_out[4] = wnz - extent_in[5];
        extent_out[5] = wnz - extent_in[4];
    }

#if defined(TECA_DEBUG)
    cerr << "extent_out=[" << extent_out[0] << ", " << extent_out[1] << ", "
        << extent_out[2] << ", " << extent_out[3] << ", " << extent_out[4]
        << ", " << extent_out[5] << "]" << endl;
#endif
}


// --------------------------------------------------------------------------
void teca_normalize_coordinates::internals_t::normalize_variables(
    bool normalize_x, bool normalize_y, bool normalize_z,
    unsigned long *extent, p_teca_array_collection data)
{
    unsigned long nx = extent[1] - extent[0] + 1;
    unsigned long ny = extent[3] - extent[2] + 1;
    unsigned long nz = extent[5] - extent[4] + 1;

    unsigned long nxy = nx*ny;
    unsigned long nxyz = nxy*nz;

    unsigned long x1 = nx - 1;
    unsigned long y1 = ny - 1;
    unsigned long z1 = nz - 1;

    unsigned int n_arrays = data->size();

    if (normalize_x)
    {
        for (unsigned int l = 0; l < n_arrays; ++l)
        {
            p_teca_variant_array a = data->get(l);
            p_teca_variant_array ao = a->new_instance(nxyz);
            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                a.get(), _A,
                NT_A *pa = static_cast<TT_A*>(a.get())->get();
                NT_A *pao = static_cast<TT_A*>(ao.get())->get();
                for (unsigned long k = 0; k < nz; ++k)
                {
                    unsigned long kk = k*nxy;
                    for (unsigned long j = 0; j < ny; ++j)
                    {
                        unsigned long jj = kk + j*nx;

                        NT_A *par = pa + jj;
                        NT_A *paor = pao + x1;

                        for (unsigned long i = 0; i < nx; ++i)
                            paor[-i] = par[i];
                    }
                }
                )
            data->set(l, ao);
        }
    }

    if (normalize_y)
    {
        unsigned int n_arrays = data->size();
        for (unsigned int l = 0; l < n_arrays; ++l)
        {
            p_teca_variant_array a = data->get(l);
            p_teca_variant_array ao = a->new_instance(nxyz);
            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                a.get(), _A,
                NT_A *pa = static_cast<TT_A*>(a.get())->get();
                NT_A *pao = static_cast<TT_A*>(ao.get())->get();
                for (unsigned long k = 0; k < nz; ++k)
                {
                    unsigned long kk = k*nxy;
                    for (unsigned long j = 0; j < ny; ++j)
                    {
                        unsigned long jj = kk + j*nx;
                        unsigned long jjo = kk + (y1 - j)*nx;
                        NT_A *par = pa + jj;
                        NT_A *paor = pao + jjo;
                        for (unsigned long i = 0; i < nx; ++i)
                            paor[i] = par[i];
                    }
                }
                )
            data->set(l, ao);
        }
    }

    if (normalize_z)
    {
        for (unsigned int l = 0; l < n_arrays; ++l)
        {
            p_teca_variant_array a = data->get(l);
            p_teca_variant_array ao = a->new_instance(nxyz);
            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                a.get(), _A,
                NT_A *pa = static_cast<TT_A*>(a.get())->get();
                NT_A *pao = static_cast<TT_A*>(ao.get())->get();
                for (unsigned long k = 0; k < nz; ++k)
                {
                    unsigned long kk = k*nxy;
                    unsigned long kko = (z1 - k)*nxy;
                    for (unsigned long j = 0; j < ny; ++j)
                    {
                        unsigned long jnx = j*nx;
                        unsigned long jj = kk + jnx;
                        unsigned long jjo = kko + jnx;

                        NT_A *par = pa + jj;
                        NT_A *paor = pao + jjo;

                        for (unsigned long i = 0; i < nx; ++i)
                            paor[i] = par[i];
                    }
                }
                )
            data->set(l, ao);
        }
    }
}

// --------------------------------------------------------------------------
teca_normalize_coordinates::teca_normalize_coordinates() : internals(nullptr)
{
    this->internals = new teca_normalize_coordinates::internals_t;

    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_normalize_coordinates::~teca_normalize_coordinates()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_normalize_coordinates::get_properties_description(
    const std::string &/*prefix*/, options_description &/*global_opts*/)
{
}

// --------------------------------------------------------------------------
void teca_normalize_coordinates::set_properties(
    const std::string &/*prefix*/, variables_map &/*opts*/)
{
}
#endif

// --------------------------------------------------------------------------
teca_metadata teca_normalize_coordinates::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_normalize_coordinates::get_output_metadata" << endl;
#endif
    (void)port;

    teca_metadata out_md(input_md[0]);

    if (!out_md.has("coordinates"))
    {
        TECA_ERROR("metadata is missing coordinates");
        return out_md;
    }

    teca_metadata coords;
    out_md.get("coordinates", coords);

    const_p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")))
    {
        TECA_ERROR("coordinates metadata is missing axes arrays")
        return out_md;
    }

    p_teca_variant_array out_x, out_y, out_z;
    if (out_x = this->internals->normalize_axis(in_x))
        coords.set("x", out_x);

    if (out_y = this->internals->normalize_axis(in_y))
        coords.set("y", out_y);

    if (out_z = this->internals->normalize_axis(in_z))
        coords.set("z", out_z);

    if (out_x || out_y || out_z)
        out_md.set("coordinates", coords);

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_normalize_coordinates::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;

    std::vector<teca_metadata> up_reqs;

    // the user will request a world or index space bounding box of data
    // transform the normalized coordinate system back to the original
    teca_metadata req(request);

    // get coordinate axes
    teca_metadata coords;
    if (input_md[0].get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing \"coordinates\"")
        return up_reqs;
    }

    p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")))
    {
        TECA_ERROR("metadata is missing coordinate arrays")
        return up_reqs;
    }

    // now convert the original coordinate axes into the
    // normalized system. this isn't cached for thread safety
    p_teca_variant_array out_x, out_y, out_z;
    out_x = this->internals->normalize_axis(in_x);
    out_y = this->internals->normalize_axis(in_y);
    out_z = this->internals->normalize_axis(in_z);

    // normalized system is the same as the original, pass the request up
    if (!out_x && !out_y && !out_z)
    {
        up_reqs.push_back(request);
        return up_reqs;
    }

    // get the original extent
    unsigned long whole_extent[6] = {0};
    if (input_md[0].get("whole_extent", whole_extent, 6))
    {
        TECA_ERROR("metadata is missing \"whole_extent\"")
        return up_reqs;
    }

    // get the extent that is being requested
    unsigned long extent_in[6] = {0};
    unsigned long extent_out[6] = {0};
    double bounds[6] = {0.0};
    if (req.get("bounds", bounds, 6))
    {
        // bounds key not present, check for extent key
        // if not present use whole_extent
        if (request.get("extent", extent_in, 6))
            memcpy(extent_in, whole_extent, 6*sizeof(unsigned long));
        else
            this->internals->normalize_extent(out_x, out_y,
                out_z, whole_extent, extent_in, extent_out);
    }
    else
    {
        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        if (teca_coordinate_util::bounds_to_extent(bounds,
            (out_x ? out_x : in_x), (out_y ? out_y : in_y),
            (out_z ? out_z : in_z), extent_in))
        {
            TECA_ERROR("invalid bounds requested.")
            return up_reqs;
        }

        this->internals->normalize_extent(out_x, out_y, out_z,
            whole_extent, extent_in, extent_out);

        // remove the bounds request, which will force the reader to
        // use the given extent
        req.remove("bounds");
    }

    req.set("extent", extent_out, 6);
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_normalize_coordinates::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_normalize_coordinates::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input mesh
    const_p_teca_cartesian_mesh in_mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    if (!in_mesh)
    {
        TECA_ERROR("Failed to compute l2 norm. dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // create output dataset, and transform data
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the original coordinate axes
    const_p_teca_variant_array in_x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array in_y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array in_z = in_mesh->get_z_coordinates();

    // normalize them
    p_teca_variant_array out_x, out_y, out_z;
    if (out_x = this->internals->normalize_axis(in_x))
        out_mesh->set_x_coordinates(out_x);

    if (out_y = this->internals->normalize_axis(in_y))
        out_mesh->set_y_coordinates(out_y);

    if (out_z = this->internals->normalize_axis(in_z))
        out_mesh->set_z_coordinates(out_z);

    // coordinates were not nomralized, nothing to do
    if (!out_x && !out_y && !out_z)
        return out_mesh;

    // fix the data
    unsigned long extent[6];
    in_mesh->get_extent(extent);

    this->internals->normalize_variables(out_x.get(),
        out_y.get(), out_z.get(), extent, out_mesh->get_point_arrays());

    // fix the extent
    /* unsigned long whole_extent[6];
    in_mesh->get_whole_extent(whole_extent);

    if (out_x)
    {
        unsigned long wnx = whole_extent[1] - whole_extent[0];
        extent[0] = wnx - extent[0];
        extent[1] = wnx - extent[1];
    }

    if (out_y)
    {
        unsigned long wny = whole_extent[3] - whole_extent[2];
        extent[2] = wny - extent[2];
        extent[3] = wny - extent[3];
    }

    if (out_y)
    {
        unsigned long wnz = whole_extent[5] - whole_extent[4];
        extent[4] = wnz - extent[4];
        extent[5] = wnz - extent[5];
    }

    out_mesh->set_extent(extent);*/

    return out_mesh;
}
