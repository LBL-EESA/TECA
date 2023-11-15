#include "teca_normalize_coordinates.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"
#include "teca_metadata_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using namespace teca_variant_array_util;

//#define TECA_DEBUG

struct internals
{
    // check and flip the axis. templated on comparison type std::less puts in
    // ascending order, std::greater puts in descending order.
    template <template<typename> typename compare_t>
    static int reorder(p_teca_variant_array &x_out,
        const const_p_teca_variant_array &x, bool &do_reorder);

    // transforms coordinates from [-180, 180] to [0, 360].
    template <typename data_t>
    static data_t periodic_shift_x(data_t x);

    // transforms coordinates from [0, 360] to [-180, 180].
    template <typename data_t>
    static data_t inv_periodic_shift_x(data_t x);

    template<typename coord_t>
    static void periodic_shift_x(coord_t *pxo,
        unsigned long *pmap, const coord_t *px, unsigned long nx);

    template<typename coord_t>
    static void inv_periodic_shift_x(unsigned long *pmap,
        const coord_t *px, unsigned long nx);

    static int inv_periodic_shift_x(
        p_teca_unsigned_long_array &map,
        const const_p_teca_variant_array &x);

    static int periodic_shift_x(p_teca_variant_array &out_x,
        p_teca_unsigned_long_array &shift_map,
        const const_p_teca_variant_array &in_x,
        bool &shifted_x);

    static int periodic_shift_x(p_teca_array_collection data,
        const teca_metadata &attributes,
        const const_p_teca_unsigned_long_array &shift_map,
        const unsigned long *extent_in,
        const unsigned long *extent_out);

    // put the y-axis in ascending order if it is not.  if a transformation was
    // applied reordered_y is set.
    static int ascending_order_y(
        p_teca_variant_array &out_y, const const_p_teca_variant_array &in_y,
        bool &reorder_y);

    // apply corresponding transformation that put the -y-axis in ascending
    // order to all data arrays
    static int ascending_order_y(p_teca_array_collection data,
        const teca_metadata &attributes, const unsigned long *extent);

    /// scale the input by fac
    static void scale(p_teca_variant_array &out,
        const const_p_teca_variant_array &in, double fac);


    template <typename data_t>
    struct indirect_less;
};



template <typename data_t>
struct internals::indirect_less
{
    indirect_less(data_t *data) : m_data(data) {}

    bool operator()(const unsigned long &l, const unsigned long &r)
    {
        return m_data[l] < m_data[r];
    }

    data_t *m_data;
};

// --------------------------------------------------------------------------
template <template<typename> typename compare_t>
int internals::reorder(p_teca_variant_array &x_out,
    const const_p_teca_variant_array &x, bool &do_reorder)
{
    unsigned long nx = x->size();
    unsigned long x1 = nx - 1;

    NESTED_VARIANT_ARRAY_DISPATCH(
        x.get(), _C,

        auto [spx, px] = get_host_accessible<CTT_C>(x);

        // if comp(x0, x1) reverse the axis.
        // when comp is less than the output will be ascending
        // when comp is greater than the output will be descending
        compare_t<NT_C> compare;
        do_reorder = compare(px[x1], px[0]);

        if (!do_reorder)
            return 0;

        using TT_C_OUT = teca_variant_array_impl<NT_C>;
        std::shared_ptr<TT_C_OUT> xo = TT_C_OUT::New(nx);
        NT_C *pxo = xo->data();

        sync_host_access_any(x);

        pxo += x1;
        for (unsigned long i = 0; i < nx; ++i)
            pxo[-i] = px[i];

        x_out = xo;

        return 0;
        )

    TECA_ERROR("Unsupported coordinate type " << x->get_class_name())
    return -1;
}

// --------------------------------------------------------------------------
template <typename data_t>
data_t internals::periodic_shift_x(data_t x)
{
#if defined(__CUDACC__)
#pragma nv_diag_suppress = unsigned_compare_with_zero
#endif
    if (x < data_t(0))
        return x + data_t(360);
    return x;
#if defined(__CUDACC__)
#pragma nv_diag_default = unsigned_compare_with_zero
#endif
}

// --------------------------------------------------------------------------
template <typename data_t>
data_t internals::inv_periodic_shift_x(data_t x)
{
    if (x > data_t(180))
        return x - data_t(360);
    return x;
}

// --------------------------------------------------------------------------
template<typename coord_t>
void internals::periodic_shift_x(coord_t *pxo,
    unsigned long *pmap, const coord_t *px, unsigned long nx)
{
    coord_t *tmp = (coord_t*)malloc(nx*sizeof(coord_t));

    // apply the periodic shift, this will leave the axis out of order
    for (unsigned long i = 0; i < nx; ++i)
        tmp[i] = periodic_shift_x(px[i]);

    // construct the map the puts the axis back into order.
    for (unsigned long i = 0; i < nx; ++i)
        pmap[i] = i;

    indirect_less<coord_t> comp(tmp);
    std::sort(pmap, pmap + nx, comp);

    // reoder the periodic shifted values
    for (unsigned long i = 0; i < nx; ++i)
        pxo[i] = tmp[pmap[i]];

    free(tmp);
}

// --------------------------------------------------------------------------
template<typename coord_t>
void internals::inv_periodic_shift_x(unsigned long *pmap,
    const coord_t *px, unsigned long nx)
{
    coord_t *tmp = (coord_t*)malloc(nx*sizeof(coord_t));

    // apply the periodic shift, this will leave the axis out of order
    for (unsigned long i = 0; i < nx; ++i)
        tmp[i] = inv_periodic_shift_x(px[i]);

    // construct the map that the puts the axis back into order.
    for (unsigned long i = 0; i < nx; ++i)
        pmap[i] = i;

    indirect_less<coord_t> comp(tmp);
    std::sort(pmap, pmap + nx, comp);

    free(tmp);
}

// --------------------------------------------------------------------------
int internals::inv_periodic_shift_x(p_teca_unsigned_long_array &map,
    const const_p_teca_variant_array &x)
{
    unsigned long nx = x->size();

    NESTED_VARIANT_ARRAY_DISPATCH(
        x.get(), _C,

        auto [spx, px] = get_host_accessible<CTT_C>(x);

        map = teca_unsigned_long_array::New(nx);
        unsigned long *pmap = map->data();

        sync_host_access_any(x);

        inv_periodic_shift_x(pmap, px, nx);

        return 0;
        )

    TECA_ERROR("Unsupported coordinate type " << x->get_class_name())
    return -1;
}

// --------------------------------------------------------------------------
int internals::periodic_shift_x(p_teca_variant_array &x_out,
    p_teca_unsigned_long_array &map, const const_p_teca_variant_array &x,
    bool &shifted_x)
{
// ignore warning about unsigned integer types. these will never need
// the periodic shift and the code will ignore them.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#if defined(__CUDACC__)
#pragma nv_diag_suppress = unsigned_compare_with_zero
#endif
    unsigned long nx = x->size();
    unsigned long x1 = nx - 1;

    NESTED_VARIANT_ARRAY_DISPATCH(
        x.get(), _C,

        auto [spx, px] = get_host_accessible<CTT_C>(x);

        sync_host_access_any(x);

        // check that the shift is needed.
        shifted_x = (px[0] < NT_C(0));

        if (!shifted_x)
            return 0;

        // this approach requires that coordinates are in ascending
        // order
        if (px[x1] < px[0])
        {
            TECA_ERROR("A periodic shift can only be apllied to"
                " coordinates in ascending order")
            return -1;
        }

        // in its current form this approach handles coordinates in
        // -180 to 180.
        if ((px[0] < NT_C(-180)) || (px[x1] > NT_C(180)))
        {
            TECA_ERROR("Invalid x-axis coordinate range ["
                << px[0] << " , " << px[x1] << "] coordinates in the"
                " range [-180.0, 180] are required")
            return -1;
        }

        // if 2 coordinate points touch the periodic boundary remnove 1 so the
        // output does not have an extranious data point at the boundary.
        if (teca_coordinate_util::equal(px[0], NT_C(-180)) &&
            teca_coordinate_util::equal(px[nx-1], NT_C(180)))
        {
            nx -= 1;
        }

        p_teca_variant_array xo = x->new_instance(nx);
        auto [pxo] = data<TT_C>(xo);

        map = teca_unsigned_long_array::New(nx);
        unsigned long *pmap = map->data();

        periodic_shift_x(pxo, pmap, px, nx);

        x_out = xo;

        return 0;
        )

    TECA_ERROR("Unsupported coordinate type " << x->get_class_name())
    return -1;
#if defined(__CUDACC__)
#pragma nv_diag_default = unsigned_compare_with_zero
#endif
#pragma GCC diagnostic pop
}

// --------------------------------------------------------------------------
int internals::ascending_order_y(
    p_teca_variant_array &out_y, const const_p_teca_variant_array &in_y,
    bool &reorder_y)
{
    if (reorder<std::less>(out_y, in_y, reorder_y))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
int internals::periodic_shift_x(p_teca_array_collection arrays,
    const teca_metadata &attributes,
    const const_p_teca_unsigned_long_array &shift_map,
    const unsigned long *extent_in,
    const unsigned long *extent_out)
{
    // apply periodic shift in the x-direction
    const unsigned long *pmap = shift_map->data();

    unsigned int n_arrays = arrays->size();
    for (unsigned int l = 0; l < n_arrays; ++l)
    {
        // get the extent of the input/output array
        const std::string &array_name = arrays->get_name(l);
        teca_metadata array_attributes;
        if (attributes.get(array_name, array_attributes))
        {
            TECA_ERROR("Failed to get the attributes for \""
                << array_name << "\"")
            return -1;
        }

        unsigned long array_extent_in[8] = {0ul};
        teca_metadata_util::get_array_extent(array_attributes,
            extent_in, array_extent_in);

        unsigned long array_extent_out[8] = {0ul};
        teca_metadata_util::get_array_extent(array_attributes,
            extent_out, array_extent_out);

        // input and output arrays may be different size if there was a duplicated
        // coordinate point on the periodic boundary
        unsigned long nxi = array_extent_in[1] - array_extent_in[0] + 1;
        unsigned long nyi = array_extent_in[3] - array_extent_in[2] + 1;
        unsigned long nxyi = nxi*nyi;

        unsigned long nxo = array_extent_out[1] - array_extent_out[0] + 1;
        unsigned long nyo = array_extent_out[3] - array_extent_out[2] + 1;
        unsigned long nzo = array_extent_out[5] - array_extent_out[4] + 1;
        unsigned long nxyo = nxo*nyo;
        unsigned long nxyzo = nxyo*nzo;

        p_teca_variant_array a = arrays->get(l);

        p_teca_variant_array ao = a->new_instance(nxyzo);
        NESTED_VARIANT_ARRAY_DISPATCH(
            a.get(), _A,

            auto [spa, pa] = get_host_accessible<CTT_A>(a);
            auto [pao] = data<TT_A>(ao);

            sync_host_access_any(a);

            for (unsigned long k = 0; k < nzo; ++k)
            {
                unsigned long kki = k*nxyi;
                unsigned long kko = k*nxyo;
                for (unsigned long j = 0; j < nyo; ++j)
                {
                    unsigned long jji = kki + j*nxi;
                    unsigned long jjo = kko + j*nxo;

                    const NT_A *par = pa + jji;
                    NT_A *paor = pao + jjo;

                    for (unsigned long i = 0; i < nxo; ++i)
                    {
                        paor[i] = par[pmap[i]];
                    }
                }
            }
            )

        arrays->set(l, ao);
    }

    return 0;
}

// --------------------------------------------------------------------------
int internals::ascending_order_y(p_teca_array_collection arrays,
    const teca_metadata &attributes, const unsigned long *mesh_extent)
{
    // for any coodinate axes that have been transformed from descending order
    // into ascending order, apply the same transform to the scalar data arrays
    unsigned int n_arrays = arrays->size();
    for (unsigned int l = 0; l < n_arrays; ++l)
    {
        const std::string &array_name = arrays->get_name(l);

        // get the extent of the array
        unsigned long array_extent[8] = {0ul};

        teca_metadata array_attributes;
        if (attributes.get(array_name, array_attributes))
        {
            TECA_ERROR("Failed to get attributes for array \""
                << array_name << "\"")
            return -1;
        }

        teca_metadata_util::get_array_extent(array_attributes,
            mesh_extent, array_extent);

        unsigned long nx = array_extent[1] - array_extent[0] + 1;
        unsigned long ny = array_extent[3] - array_extent[2] + 1;
        unsigned long nz = array_extent[5] - array_extent[4] + 1;

        unsigned long nxy = nx*ny;
        unsigned long nxyz = nxy*nz;

        unsigned long y1 = ny - 1;

        // apply the transform
        p_teca_variant_array a = arrays->get(l);
        p_teca_variant_array ao = a->new_instance(nxyz);
        NESTED_VARIANT_ARRAY_DISPATCH(
            a.get(), _A,

            auto [spa, pa] = get_host_accessible<CTT_A>(a);
            auto [pao] = ::data<TT_A>(ao);

            sync_host_access_any(a);

            for (unsigned long k = 0; k < nz; ++k)
            {
                unsigned long kk = k*nxy;
                for (unsigned long j = 0; j < ny; ++j)
                {
                    unsigned long jj = kk + j*nx;
                    unsigned long jjo = kk + (y1 - j)*nx;
                    const NT_A *par = pa + jj;
                    NT_A *paor = pao + jjo;
                    for (unsigned long i = 0; i < nx; ++i)
                        paor[i] = par[i];
                }
            }
            )
        arrays->set(l, ao);
    }

    return 0;
}

// --------------------------------------------------------------------------
void internals::scale(p_teca_variant_array &out,
    const const_p_teca_variant_array &in, double fac)
{
    unsigned long n_elem = in->size();

    VARIANT_ARRAY_DISPATCH(in.get(),
        NT *pout = nullptr;
        std::tie(out, pout) = New<TT>(n_elem);

        auto [spin, pin] = get_host_accessible<CTT>(in);

        sync_host_access_any(in);

        for (unsigned long i = 0; i < n_elem; ++i)
            pout[i] = pin[i] * fac;
        )
}



// --------------------------------------------------------------------------
teca_normalize_coordinates::teca_normalize_coordinates() :
    enable_periodic_shift_x(0), enable_y_axis_ascending(1),
    enable_unit_conversions(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_normalize_coordinates::~teca_normalize_coordinates()
{
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_normalize_coordinates::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_normalize_coordinates":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, enable_periodic_shift_x,
            "Enables application of periodic shift in the x-direction.")
        TECA_POPTS_GET(int, prefix, enable_y_axis_ascending,
            "Enables transformtion the ensures the y-axis is in"
            " ascending order.")
        TECA_POPTS_GET(int, prefix, enable_unit_conversions,
            "Enables conversions of the axis units")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_normalize_coordinates::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, enable_periodic_shift_x)
    TECA_POPTS_SET(opts, int, prefix, enable_y_axis_ascending)
    TECA_POPTS_SET(opts, int, prefix, enable_unit_conversions)
}
#endif

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
        TECA_FATAL_ERROR("metadata is missing coordinates");
        return teca_metadata();
    }

    const_p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")))
    {
        TECA_FATAL_ERROR("coordinates metadata is missing axes arrays")
        return teca_metadata();
    }

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_FATAL_ERROR("Failed to apply periodic shift to the x-axis")
        return teca_metadata();
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_FATAL_ERROR("Failed to put the y-axis in ascending order")
        return teca_metadata();
    }

    // check for units of Pa on the z-axis
    bool z_axis_unit_conv = false;
    double z_axis_conv_fac = 1.0;
    p_teca_variant_array out_z;
    if (this->enable_unit_conversions)
    {
        teca_metadata attributes;
        std::string z_variable;
        teca_metadata z_attributes;
        std::string z_units;
        if (coords.get("z_variable", z_variable) ||
            out_md.get("attributes", attributes) ||
            attributes.get(z_variable, z_attributes) ||
            z_attributes.get("units", z_units))
        {
            TECA_FATAL_ERROR("Units check failed. Failed to get"
                " z-axis attributes or units")
            return teca_metadata();
        }

        if ((z_units == "hPa") || (z_units.compare(0,8,"millibar") == 0))
        {
            z_axis_unit_conv = true;
            z_axis_conv_fac = 100.0;

            internals::scale(out_z, in_z, z_axis_conv_fac);

            z_attributes.set("units", std::string("Pa"));
            attributes.set(z_variable, z_attributes);

            out_md.set("attributes", attributes);
        }
        else if (z_units != "Pa")
        {
            TECA_FATAL_ERROR("z-axis has unsupported units \""
                << z_units << "\"")
            return teca_metadata();
        }
    }

    // pass normalized coordinates
    if (out_x)
    {
        coords.set("x", out_x);

        // update the whole extent in case the coordinate axis touches the periodic boundary
        if (out_x->size() != in_x->size())
        {
            unsigned long whole_extent[6];
            if (out_md.get("whole_extent", whole_extent, 6))
            {
                TECA_FATAL_ERROR("Failed to get the input whole_extent")
                return teca_metadata();
            }
            whole_extent[1] -= 1;
            out_md.set("whole_extent", whole_extent);
        }
    }

    if (out_y)
        coords.set("y", out_y);

    if (out_z)
        coords.set("z", out_z);

    if (out_x || out_y || out_z)
    {
        double bounds[6] = {0.0};
        teca_coordinate_util::get_cartesian_mesh_bounds(
            out_x ? out_x : in_x, out_y ? out_y : in_y, out_z ? out_z : in_z,
            bounds);
        out_md.set("coordinates", coords);
        out_md.set("bounds", bounds);
    }

    if ((this->verbose > 1) &&
        teca_mpi_util::mpi_rank_0(this->get_communicator()))
    {
        if (reordered_y)
            TECA_STATUS("The y-axis will be transformed to be in ascending order.")

        if (shifted_x)
            TECA_STATUS("The x-axis will be transformed from [-180, 180] to [0, 360].")

        if (z_axis_unit_conv)
            TECA_STATUS("The z-axis units were scaled by " << z_axis_conv_fac)
    }

    return out_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_normalize_coordinates::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_normalize_coordinates::get_upstream_request" << std::endl;
#endif
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
        TECA_FATAL_ERROR("metadata is missing \"coordinates\"")
        return up_reqs;
    }

    p_teca_variant_array in_x, in_y, in_z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(in_z = coords.get("z")))
    {
        TECA_FATAL_ERROR("metadata is missing coordinate arrays")
        return up_reqs;
    }

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_FATAL_ERROR("Failed to apply periodic shift to the x-axis")
        return up_reqs;
    }

    // compute the inverse map
    p_teca_unsigned_long_array inv_shift_map;
    if (shifted_x && internals::inv_periodic_shift_x(inv_shift_map, out_x))
    {
        TECA_FATAL_ERROR("Failed to compute the inverse shift map")
        return up_reqs;
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_FATAL_ERROR("Failed to put the y-axis in ascending order")
        return up_reqs;
    }

    // check for units of Pa on the z-axis
    bool z_axis_unit_conv = false;
    double z_axis_conv_fac = 1.0;
    p_teca_variant_array out_z;
    if (this->enable_unit_conversions)
    {
        teca_metadata attributes;
        std::string z_variable;
        teca_metadata z_attributes;
        std::string z_units;

        coords.get("z_variable", z_variable);
        input_md[0].get("attributes", attributes);
        attributes.get(z_variable, z_attributes);
        z_attributes.get("units", z_units);

        if ((z_units == "hPa") || (z_units.compare(0,8,"millibar") == 0))
        {
            z_axis_unit_conv = true;
            z_axis_conv_fac = 100.0;
            internals::scale(out_z, in_z, z_axis_conv_fac);
        }
    }

    // get the transformed bounds
    const_p_teca_variant_array x = out_x ? out_x : in_x;
    const_p_teca_variant_array y = out_y ? out_y : in_y;
    const_p_teca_variant_array z = out_z ? out_z : in_z;

    double bounds[6] = {0.0};
    teca_coordinate_util::get_cartesian_mesh_bounds(x, y, z, bounds);

    // get the original extent
    unsigned long whole_extent[6] = {0};
    if (input_md[0].get("whole_extent", whole_extent, 6))
    {
        TECA_FATAL_ERROR("metadata is missing \"whole_extent\"")
        return up_reqs;
    }

    // get the extent that is being requested
    unsigned long extent_in[6] = {0};
    double req_bounds[6] = {0.0};
    if (req.get("bounds", req_bounds, 6))
    {
        // bounds key not present, check for extent key if not present use
        // whole_extent
        if (request.get("extent", extent_in, 6))
        {
            // correct in case we removed a duplicated point at the periodic
            // boundary
            if (out_x && (in_x->size() != out_x->size()))
                whole_extent[1]  -= 1;

            memcpy(extent_in, whole_extent, 6*sizeof(unsigned long));
        }

        // convert extent to bounds
        x->get(extent_in[0], req_bounds[0]);
        x->get(extent_in[1], req_bounds[1]);
        y->get(extent_in[2], req_bounds[2]);
        y->get(extent_in[3], req_bounds[3]);
        z->get(extent_in[4], req_bounds[4]);
        z->get(extent_in[5], req_bounds[5]);
    }
    else
    {
        // remove the bounds request, which will force the reader to
        // use the given extent
        req.remove("bounds");
    }

    // validate the requested bounds
    if (!teca_coordinate_util::same_orientation(bounds, req_bounds) ||
        !teca_coordinate_util::covers(bounds, req_bounds))
    {
        TECA_FATAL_ERROR("Invalid request. The requested bounds ["
            <<  req_bounds[0] << ", " << req_bounds[1] << ", "
            <<  req_bounds[2] << ", " << req_bounds[3] << ", "
            <<  req_bounds[4] << ", " << req_bounds[5]
            << "] is not covered by the available bounds ["
            <<  bounds[0] << ", " << bounds[1] << ", "
            <<  bounds[2] << ", " << bounds[3] << ", "
            <<  bounds[4] << ", " << bounds[5] << "]")
        return up_reqs;
    }

    // transform the bounds
    double tfm_bounds[6];
    memcpy(tfm_bounds, req_bounds, 6*sizeof(double));

    if (shifted_x)
    {
        // if a bounds request crosses the periodic boundary
        // then it needs to be split into 2 requests. Eg: a request
        // for [90, 270] becomes [-180, -90], [90, 180].
        // If this comes up a lot we could implement as follows:
        // issue both requests here and merge the data in execute.

        if (((req_bounds[0] < 180.0) && (req_bounds[1] > 180.0)) ||
            teca_coordinate_util::equal(req_bounds[0], 180.0) ||
            teca_coordinate_util::equal(req_bounds[1], 180.0))
        {
            // crosses periodic boundary (TODO handle subseting)
            unsigned long x1 = in_x->size() - 1;
            in_x->get(0, tfm_bounds[0]);
            in_x->get(x1, tfm_bounds[1]);

            TECA_WARNING("The requested x-axis bounds"
                " [" << req_bounds[0] << ", " << req_bounds[1] << "] cross a"
                " periodic boundary. Subsetting across a periodic boundary is"
                " not fully implemented. Requesting the entire x-axis ["
                << tfm_bounds[0] << ", " << tfm_bounds[1] << "]")
        }
        else
        {
            tfm_bounds[0] = internals::inv_periodic_shift_x(tfm_bounds[0]);
            tfm_bounds[1] = internals::inv_periodic_shift_x(tfm_bounds[1]);
        }
    }

    if (reordered_y)
    {
        std::swap(tfm_bounds[2], tfm_bounds[3]);
    }

    if (z_axis_unit_conv)
    {
        tfm_bounds[4] /= z_axis_conv_fac;
        tfm_bounds[5] /= z_axis_conv_fac;
    }

    // convert the transformed bounds to an
    // an extent that covers them in the upstream coordinate system
    unsigned long extent_out[6];
    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    if (teca_coordinate_util::bounds_to_extent(tfm_bounds,
            in_x, in_y, in_z, extent_out) ||
        teca_coordinate_util::validate_extent(in_x->size(),
            in_y->size(), in_z->size(), extent_out, true))
    {
        TECA_FATAL_ERROR("invalid bounds requested.")
        return up_reqs;
    }

#ifdef TECA_DEBUG
    std::cerr << "req_bounds = [" <<  req_bounds[0]
        << ", " << req_bounds[1] << ", " <<  req_bounds[2]
        << ", " << req_bounds[3] << ", " <<  req_bounds[4]
        << ", " << req_bounds[5] << "]" << std::endl
        << "tfm_bounds = [" <<  tfm_bounds[0]
        << ", " << tfm_bounds[1] << ", " <<  tfm_bounds[2]
        << ", " << tfm_bounds[3] << ", " <<  tfm_bounds[4]
        << ", " << tfm_bounds[5] << "]" << std::endl
        << "extent_in = [" <<  extent_in[0]
        << ", " << extent_in[1] << ", " <<  extent_in[2]
        << ", " << extent_in[3] << ", " <<  extent_in[4]
        << ", " << extent_in[5] << "]" << std::endl
        << "extent_out = [" <<  extent_out[0]
        << ", " << extent_out[1] << ", " <<  extent_out[2]
        << ", " << extent_out[3] << ", " <<  extent_out[4]
        << ", " << extent_out[5] << "]" << std::endl;
#endif

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
        TECA_FATAL_ERROR("The input dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // create output dataset, and transform data
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the original coordinate axes, these may be in descending order
    const_p_teca_variant_array in_x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array in_y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array in_z = in_mesh->get_z_coordinates();

    // get the whole extent
    unsigned long whole_extent_in[6];
    in_mesh->get_whole_extent(whole_extent_in);

    unsigned long whole_extent_out[6];
    memcpy(whole_extent_out, whole_extent_in, 6*sizeof(unsigned long));

    // get the extent
    unsigned long extent_in[6];
    in_mesh->get_extent(extent_in);

    unsigned long extent_out[6];
    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    // get the bounds
    double bounds_in[6];
    in_mesh->get_bounds(bounds_in);

    double bounds_out[6];
    memcpy(bounds_out, bounds_in, 6*sizeof(double));

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_FATAL_ERROR("Failed to apply periodic shift to the x-axis")
        return nullptr;
    }

    teca_metadata attributes;

    if (shifted_x)
    {
        in_mesh->get_attributes(attributes);

        if (this->verbose &&
            teca_mpi_util::mpi_rank_0(this->get_communicator()))
        {
            TECA_STATUS("The x-axis will be transformed from [-180, 180] to [0, 360].")
        }

        // update the mesh coordinates
        std::string var;
        in_mesh->get_x_coordinate_variable(var);
        out_mesh->set_x_coordinates(var, out_x);

        // correct extent in case the coordinate axis touches the periodic boundary
        if (out_x && (out_x->size() != in_x->size()))
        {
            if (teca_mpi_util::mpi_rank_0(this->get_communicator()))
            {
                TECA_WARNING("The coordinate and data on the periodic boundary"
                    " at x = +/- 180 is duplicated.")
            }

            extent_out[1] -= 1;
            whole_extent_out[1] -= 1;
        }

        // correct the bounds
        out_x->get(extent_out[0], bounds_out[0]);
        out_x->get(extent_out[1], bounds_out[1]);

        // shift the data arrays
        if (internals::periodic_shift_x(out_mesh->get_point_arrays(),
            attributes, shift_map, extent_in, extent_out))
        {
            TECA_FATAL_ERROR("Failed to apply periodic shift in the x direction")
            return nullptr;
        }
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_FATAL_ERROR("Failed to put the y-axis in ascending order")
        return nullptr;
    }

    if (reordered_y)
    {
        if (attributes.empty())
            in_mesh->get_attributes(attributes);

        if (this->verbose &&
            teca_mpi_util::mpi_rank_0(this->get_communicator()))
        {
            TECA_STATUS("The y-axis will be transformed to be in ascending order.")
        }

        // update the mesh coordinates
        std::string var;
        in_mesh->get_y_coordinate_variable(var);
        out_mesh->set_y_coordinates(var, out_y);

        // reorder the data arrays
        if (internals::ascending_order_y(out_mesh->get_point_arrays(),
            attributes, extent_out))
        {
            TECA_FATAL_ERROR("Failed to put point arrays into ascending order")
            return nullptr;
        }

        // flip the y axis extent
        unsigned long whole_ny = whole_extent_in[3] - whole_extent_in[2] + 1;
        extent_out[2] = whole_ny - extent_in[3] - 1;
        extent_out[3] = whole_ny - extent_in[2] - 1;

        // flip the y-axis bounds
        bounds_out[2] = bounds_in[3];
        bounds_out[3] = bounds_in[2];
    }

    // check for units of Pa on the z-axis
    bool z_axis_unit_conv = false;
    double z_axis_conv_fac = 1.0;
    p_teca_variant_array out_z;
    if (this->enable_unit_conversions)
    {
        if (attributes.empty())
            in_mesh->get_attributes(attributes);

        std::string z_var, z_units;
        teca_metadata z_attributes;

        in_mesh->get_z_coordinate_variable(z_var);
        attributes.get(z_var, z_attributes);
        z_attributes.get("units", z_units);

        if ((z_units == "hPa") || (z_units.compare(0,8,"millibar") == 0))
        {
            z_axis_unit_conv = true;
            z_axis_conv_fac = 100.0;
            internals::scale(out_z, in_z, z_axis_conv_fac);

            z_attributes.set("units", std::string("Pa"));
            attributes.set(z_var, z_attributes);

            out_mesh->set_attributes(attributes);
            out_mesh->set_z_coordinates(z_var, out_z);

            bounds_out[4] *= z_axis_conv_fac;
            bounds_out[5] *= z_axis_conv_fac;
        }

        if (this->verbose && z_axis_unit_conv &&
            teca_mpi_util::mpi_rank_0(this->get_communicator()))
        {
            TECA_STATUS("The z-axis units were scaled by " << z_axis_conv_fac)
        }
    }

    // update the mesh extent
    if (shifted_x || reordered_y || z_axis_unit_conv)
    {
        out_mesh->set_extent(extent_out);
        out_mesh->set_whole_extent(whole_extent_out);
        out_mesh->set_bounds(bounds_out);
    }

    return out_mesh;
}
