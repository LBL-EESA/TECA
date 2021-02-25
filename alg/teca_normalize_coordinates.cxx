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

//#define TECA_DEBUG

struct teca_normalize_coordinates::internals_t
{
    internals_t() {}
    ~internals_t() {}

    static
    int normalize_axes(
        p_teca_variant_array &out_x,
        p_teca_variant_array &out_y,
        p_teca_variant_array &out_z,
        const const_p_teca_variant_array &in_x,
        const const_p_teca_variant_array &in_y,
        const const_p_teca_variant_array &in_z,
        int x_axis_order, int y_axis_order,
        int z_axis_order, double *bounds);

    template <template<typename> typename compare_t>
    static p_teca_variant_array normalize_axis(
        const const_p_teca_variant_array &x, double *bounds);

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
template <template<typename> typename compare_t>
p_teca_variant_array teca_normalize_coordinates::internals_t::normalize_axis(
    const const_p_teca_variant_array &x, double *bounds)
{
    unsigned long nx = x->size();
    unsigned long x1 = nx - 1;

    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        x.get(), _C,

        const NT_C *px = dynamic_cast<TT_C*>(x.get())->get();

        // if comp(x0, x1) reverse the axis.
        // when comp is less than the output will be ascending
        // when comp is greater than the output will be descending
        compare_t compare;
        if (compare(px[x1], px[0]))
        {
            p_teca_variant_array xo = x->new_instance(nx);
            NT_C *pxo = static_cast<teca_variant_array_impl<NT_C>*>(xo.get())->get();

            pxo += x1;
            for (unsigned long i = 0; i < nx; ++i)
                pxo[-i] = px[i];

            bounds[0] = px[x1];
            bounds[1] = px[0];

            return xo;
        }

        bounds[0] = px[0];
        bounds[1] = px[x1];
        )
    return nullptr;
}

// --------------------------------------------------------------------------
int teca_normalize_coordinates::internals_t::normalize_axes(
    p_teca_variant_array &out_x,
    p_teca_variant_array &out_y,
    p_teca_variant_array &out_z,
    const const_p_teca_variant_array &in_x,
    const const_p_teca_variant_array &in_y,
    const const_p_teca_variant_array &in_z,
    int x_axis_order, int y_axis_order, int z_axis_order,
    double *bounds)
{
    // x axis
    if (x_axis_order == ORDER_ASCENDING)
    {
       out_x = internals_t::normalize_axis<std::less>(in_x, bounds);
    }
    else if (x_axis_order == ORDER_DESCENDING)
    {
       out_x = internals_t::normalize_axis<std::greater>(in_x, bounds);
    }
    else
    {
        TECA_ERROR("Invalid x_axis_order " << x_axis_order)
        return -1;
    }

    // y axis
    if (y_axis_order == ORDER_ASCENDING)
    {
       out_y = internals_t::normalize_axis<std::less>(in_y, bounds + 2);
    }
    else if (y_axis_order == ORDER_DESCENDING)
    {
       out_y = internals_t::normalize_axis<std::greater>(in_y, bounds + 2);
    }
    else
    {
        TECA_ERROR("Invalid y_axis_order " << y_axis_order)
        return -1;
    }

    // z axis
    if (z_axis_order == ORDER_ASCENDING)
    {
       out_z = internals_t::normalize_axis<std::less>(in_z, bounds + 4);
    }
    else if (z_axis_order == ORDER_DESCENDING)
    {
       out_z = internals_t::normalize_axis<std::greater>(in_z, bounds + 4);
    }
    else
    {
        TECA_ERROR("Invalid z_axis_order " << z_axis_order)
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
void teca_normalize_coordinates::internals_t::normalize_extent(
    p_teca_variant_array out_x, p_teca_variant_array out_y,
    p_teca_variant_array out_z, unsigned long *whole_extent,
    unsigned long *extent_in, unsigned long *extent_out)
{
#if defined(TECA_DEBUG)
    std::cerr
        << "out=[" << out_x << ", " << out_y << ", " << out_z << "]" << std::endl
        << "whole_extent=[" << whole_extent[0] << ", " << whole_extent[1] << ", "
        << whole_extent[2] << ", " << whole_extent[3] << ", " << whole_extent[4]
        << ", " << whole_extent[5] << "]" << std::endl
        << "extent_in=[" << extent_in[0] << ", " << extent_in[1] << ", "
        << extent_in[2] << ", " << extent_in[3] << ", " << extent_in[4] << ", "
        << extent_in[5] << "]" << std::endl;
#endif

    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    // detect coordinate axes in descending order, transform the incoming
    // extents from ascending order coordinates back to original descending
    // order coordinate system so the upstream gets the correct extent
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
    std::cerr << "extent_out=[" << extent_out[0] << ", " << extent_out[1] << ", "
        << extent_out[2] << ", " << extent_out[3] << ", " << extent_out[4]
        << ", " << extent_out[5] << "]" << std::endl;
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

    // for any coodinate axes that have been transformed from descending order
    // into ascending order, apply the same transform to the scalar data arrays
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
                        NT_A *paor = pao + jj + x1;

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
teca_normalize_coordinates::teca_normalize_coordinates() :
    x_axis_order(ORDER_ASCENDING), y_axis_order(ORDER_ASCENDING),
    z_axis_order(ORDER_DESCENDING), internals(nullptr)
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
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_normalize_coordinates":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, x_axis_order,
            "Sets the desired output order of the x-axis. Use"
            " ORDER_ASCENDING(0) or ORDER_DESCENDING(1).")
        TECA_POPTS_GET(int, prefix, y_axis_order,
            "Sets the desired output order of the y-axis. Use"
            " ORDER_ASCENDING(0) or ORDER_DESCENDING(1).")
        TECA_POPTS_GET(int, prefix, z_axis_order,
            "Sets the desired output order of the z-axis. Use"
            " ORDER_ASCENDING(0) or ORDER_DESCENDING(1).")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_normalize_coordinates::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, x_axis_order)
    TECA_POPTS_SET(opts, int, prefix, y_axis_order)
    TECA_POPTS_SET(opts, int, prefix, z_axis_order)
}
#endif

// --------------------------------------------------------------------------
int teca_normalize_coordinates::validate_x_axis_order(int val)
{
    if ((val != ORDER_ASCENDING) && (val != ORDER_DESCENDING))
    {
        TECA_ERROR("Invlaid x_axis_order " << val)
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_normalize_coordinates::validate_y_axis_order(int val)
{
    if ((val != ORDER_ASCENDING) && (val != ORDER_DESCENDING))
    {
        TECA_ERROR("Invlaid y_axis_order " << val)
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_normalize_coordinates::validate_z_axis_order(int val)
{
    if ((val != ORDER_ASCENDING) && (val != ORDER_DESCENDING))
    {
        TECA_ERROR("Invlaid z_axis_order " << val)
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
teca_metadata teca_normalize_coordinates::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_normalize_coordinates::get_output_metadata" << std::endl;
#endif
    (void)port;

    teca_metadata out_md(input_md[0]);

    teca_metadata coords;
    if (out_md.get("coordinates", coords))
    {
        TECA_ERROR("metadata is missing coordinates");
        return teca_metadata();
    }

    const_p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")))
    {
        TECA_ERROR("coordinates metadata is missing axes arrays")
        return teca_metadata();
    }

    // check for and transform coordinate axes from descending order
    // to ascending order
    double bounds[6] = {0.0};
    p_teca_variant_array out_x, out_y, out_z;
    if (this->internals->normalize_axes(out_x, out_y, out_z,
        in_x, in_y, in_z, this->x_axis_order, this->y_axis_order,
        this->z_axis_order, bounds))
    {
        TECA_ERROR("Failed to normalize axes")
        return teca_metadata();
    }

    // pass normalized coordinates
    if (out_x)
        coords.set("x", out_x);

    if (out_y)
        coords.set("y", out_y);

    if (out_z)
        coords.set("z", out_z);

    if (out_x || out_y || out_z)
    {
        out_md.set("coordinates", coords);
        out_md.set("bounds", bounds);
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_normalize_coordinates::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;

    std::vector<teca_metadata> up_reqs;

    // down stream requests of a world or index space bounding box of data are
    // always in ascending order coordinate system if the upstream is providing
    // data in a descending order cooridnate system transform the request into
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
    double bounds[6] = {0.0};
    p_teca_variant_array out_x, out_y, out_z;
    if (this->internals->normalize_axes(out_x, out_y, out_z,
        in_x, in_y, in_z, this->x_axis_order, this->y_axis_order,
        this->z_axis_order, bounds))
    {
        TECA_ERROR("Failed to normalize axes")
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
    double req_bounds[6] = {0.0};
    if (req.get("bounds", req_bounds, 6))
    {
        // bounds key not present, check for extent key
        // if not present use whole_extent
        if (request.get("extent", extent_in, 6))
            memcpy(extent_in, whole_extent, 6*sizeof(unsigned long));
    }
    else
    {
        // validate the requested bounds
        if (!teca_coordinate_util::same_orientation(bounds, req_bounds) ||
            !teca_coordinate_util::covers(bounds, req_bounds))
        {
            TECA_ERROR("Invalid request. The requested bounds ["
                <<  req_bounds[0] << ", " << req_bounds[1] << ", "
                <<  req_bounds[2] << ", " << req_bounds[3] << ", "
                <<  req_bounds[4] << ", " << req_bounds[5]
                << "] is not covered by the available bounds ["
                <<  bounds[0] << ", " << bounds[1] << ", "
                <<  bounds[2] << ", " << bounds[3] << ", "
                <<  bounds[4] << ", " << bounds[5] << "]")
            return up_reqs;
        }

        // bounds key was present, convert the bounds to an
        // an extent that covers them.
        if (teca_coordinate_util::bounds_to_extent(req_bounds,
            (out_x ? out_x : in_x), (out_y ? out_y : in_y),
            (out_z ? out_z : in_z), extent_in))
        {
            TECA_ERROR("invalid bounds requested.")
            return up_reqs;
        }

        // remove the bounds request, which will force the reader to
        // use the given extent
        req.remove("bounds");
    }

    // apply the transform if needed
    this->internals->normalize_extent(out_x, out_y, out_z,
        whole_extent, extent_in, extent_out);

    // validate the requested extent
    if (!teca_coordinate_util::covers_ascending(whole_extent, extent_out))
    {
        TECA_ERROR("Invalid request. The requested extent ["
            <<  extent_out[0] << ", " << extent_out[1] << ", "
            <<  extent_out[2] << ", " << extent_out[3] << ", "
            <<  extent_out[4] << ", " << extent_out[5]
            << "] is not covered by the available whole_extent ["
            <<  whole_extent[0] << ", " << whole_extent[1] << ", "
            <<  whole_extent[2] << ", " << whole_extent[3] << ", "
            <<  whole_extent[4] << ", " << whole_extent[5] << "]")
        return up_reqs;
    }

    // send the request up
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
    std::cerr << teca_parallel_id()
        << "teca_normalize_coordinates::execute" << std::endl;
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

    // get the original coordinate axes, these may be in descending order
    const_p_teca_variant_array in_x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array in_y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array in_z = in_mesh->get_z_coordinates();

    // transform the axes to ascending order if needed
    double bounds[6] = {0.0};
    p_teca_variant_array out_x, out_y, out_z;
    if (this->internals->normalize_axes(out_x, out_y, out_z,
        in_x, in_y, in_z, this->x_axis_order, this->y_axis_order,
        this->z_axis_order, bounds))
    {
        TECA_ERROR("Failed to normalize axes")
        return nullptr;
    }

    if (out_x)
    {
        std::string var;
        in_mesh->get_x_coordinate_variable(var);
        out_mesh->set_x_coordinates(var, out_x);
    }

    if (out_y)
    {
        std::string var;
        in_mesh->get_y_coordinate_variable(var);
        out_mesh->set_y_coordinates(var, out_y);
    }

    if (out_z)
    {
        std::string var;
        in_mesh->get_z_coordinate_variable(var);
        out_mesh->set_z_coordinates(var, out_z);
    }

    // apply the same set of transforms to the data
    if (out_x || out_y || out_z)
    {
        unsigned long extent[6];
        in_mesh->get_extent(extent);

        this->internals->normalize_variables(out_x.get(),
            out_y.get(), out_z.get(), extent,
            out_mesh->get_point_arrays());
    }

    return out_mesh;
}
