#include "teca_normalize_coordinates.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
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

    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        x.get(), _C,

        const NT_C *px = dynamic_cast<TT_C*>(x.get())->get();

        // if comp(x0, x1) reverse the axis.
        // when comp is less than the output will be ascending
        // when comp is greater than the output will be descending
        compare_t compare;
        do_reorder = compare(px[x1], px[0]);

        if (!do_reorder)
            return 0;

        p_teca_variant_array xo = x->new_instance(nx);
        NT_C *pxo = static_cast<teca_variant_array_impl<NT_C>*>(xo.get())->get();

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
    if (x < data_t(0))
        return x + data_t(360);
    return x;
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

    indirect_less comp(tmp);
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

    indirect_less comp(tmp);
    std::sort(pmap, pmap + nx, comp);

    free(tmp);
}

// --------------------------------------------------------------------------
int internals::inv_periodic_shift_x(p_teca_unsigned_long_array &map,
    const const_p_teca_variant_array &x)
{
    unsigned long nx = x->size();

    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        x.get(), _C,

        const NT_C *px = dynamic_cast<TT_C*>(x.get())->get();

        map = teca_unsigned_long_array::New(nx);
        unsigned long *pmap = map->get();

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
    unsigned long nx = x->size();
    unsigned long x1 = nx - 1;

    NESTED_TEMPLATE_DISPATCH(const teca_variant_array_impl,
        x.get(), _C,

        const NT_C *px = dynamic_cast<TT_C*>(x.get())->get();

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
        NT_C *pxo = static_cast<teca_variant_array_impl<NT_C>*>
            (xo.get())->get();

        map = teca_unsigned_long_array::New(nx);
        unsigned long *pmap = map->get();

        periodic_shift_x(pxo, pmap, px, nx);

        x_out = xo;

        return 0;
        )

    TECA_ERROR("Unsupported coordinate type " << x->get_class_name())
    return -1;
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
int internals::periodic_shift_x(p_teca_array_collection data,
    const teca_metadata &attributes,
    const const_p_teca_unsigned_long_array &shift_map,
    const unsigned long *extent_in,
    const unsigned long *extent_out)
{
    // apply periodic shift in the x-direction
    const unsigned long *pmap = shift_map->get();

    unsigned int n_arrays = data->size();
    for (unsigned int l = 0; l < n_arrays; ++l)
    {
        // get the extent of the input/output array
        const std::string &array_name = data->get_name(l);
        teca_metadata array_attributes;
        if (attributes.get(array_name, array_attributes))
        {
            TECA_ERROR("Failed to get the attributes for \""
                << array_name << "\"")
            return -1;
        }

        unsigned long array_extent_in[6] = {0ul};
        teca_metadata_util::get_array_extent(array_attributes,
            extent_in, array_extent_in);

        unsigned long array_extent_out[6] = {0ul};
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

        p_teca_variant_array a = data->get(l);

        p_teca_variant_array ao = a->new_instance(nxyzo);
        NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
            a.get(), _A,
            NT_A *pa = static_cast<TT_A*>(a.get())->get();
            NT_A *pao = static_cast<TT_A*>(ao.get())->get();
            for (unsigned long k = 0; k < nzo; ++k)
            {
                unsigned long kki = k*nxyi;
                unsigned long kko = k*nxyo;
                for (unsigned long j = 0; j < nyo; ++j)
                {
                    unsigned long jji = kki + j*nxi;
                    unsigned long jjo = kko + j*nxo;

                    NT_A *par = pa + jji;
                    NT_A *paor = pao + jjo;

                    for (unsigned long i = 0; i < nxo; ++i)
                    {
                        paor[i] = par[pmap[i]];
                    }
                }
            }
            )

        data->set(l, ao);
    }

    return 0;
}

// --------------------------------------------------------------------------
int internals::ascending_order_y(p_teca_array_collection data,
    const teca_metadata &attributes, const unsigned long *mesh_extent)
{
    // for any coodinate axes that have been transformed from descending order
    // into ascending order, apply the same transform to the scalar data arrays
    unsigned int n_arrays = data->size();
    for (unsigned int l = 0; l < n_arrays; ++l)
    {
        const std::string &array_name = data->get_name(l);

        // get the extent of the array
        unsigned long array_extent[6] = {0ul};

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

    return 0;
}



// --------------------------------------------------------------------------
teca_normalize_coordinates::teca_normalize_coordinates() :
    enable_periodic_shift_x(0), enable_y_axis_ascending(1)
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

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_ERROR("Failed to apply periodic shift to the x-axis")
        return teca_metadata();
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_ERROR("Failed to put the y-axis in ascending order")
        return teca_metadata();
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
                TECA_ERROR("Failed to get the input whole_extent")
                return teca_metadata();
            }
            whole_extent[1] -= 1;
            out_md.set("whole_extent", whole_extent);
        }
    }

    if (out_y)
        coords.set("y", out_y);

    if (out_x || out_y)
    {
        double bounds[6] = {0.0};
        teca_coordinate_util::get_cartesian_mesh_bounds(
            out_x ? out_x : in_x, out_y ? out_y : in_y, in_z,
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
        TECA_ERROR("metadata is missing \"coordinates\"")
        return up_reqs;
    }

    p_teca_variant_array in_x, in_y, z;
    if (!(in_x = coords.get("x")) || !(in_y = coords.get("y"))
        || !(z = coords.get("z")))
    {
        TECA_ERROR("metadata is missing coordinate arrays")
        return up_reqs;
    }

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_ERROR("Failed to apply periodic shift to the x-axis")
        return up_reqs;
    }

    // compute the inverse map
    p_teca_unsigned_long_array inv_shift_map;
    if (shifted_x && internals::inv_periodic_shift_x(inv_shift_map, out_x))
    {
        TECA_ERROR("Failed to compute the inverse shifty map")
        return up_reqs;
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_ERROR("Failed to put the y-axis in ascending order")
        return up_reqs;
    }

    // get the transformed bounds
    const_p_teca_variant_array x = out_x ? out_x : in_x;
    const_p_teca_variant_array y = out_y ? out_y : in_y;

    double bounds[6] = {0.0};
    teca_coordinate_util::get_cartesian_mesh_bounds(x, y, z, bounds);

    // get the original extent
    unsigned long whole_extent[6] = {0};
    if (input_md[0].get("whole_extent", whole_extent, 6))
    {
        TECA_ERROR("metadata is missing \"whole_extent\"")
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

    // convert the transformed bounds to an
    // an extent that covers them in the upstream coordinate system
    unsigned long extent_out[6];
    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    if (teca_coordinate_util::bounds_to_extent(tfm_bounds,
            in_x, in_y, z, extent_out) ||
        teca_coordinate_util::validate_extent(in_x->size(),
            in_y->size(), z->size(), extent_out, true))
    {
        TECA_ERROR("invalid bounds requested.")
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
        TECA_ERROR("The input dataset is not a teca_cartesian_mesh")
        return nullptr;
    }

    // create output dataset, and transform data
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();
    out_mesh->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the original coordinate axes, these may be in descending order
    const_p_teca_variant_array in_x = in_mesh->get_x_coordinates();
    const_p_teca_variant_array in_y = in_mesh->get_y_coordinates();
    const_p_teca_variant_array in_z = in_mesh->get_z_coordinates();

    // get the extent
    unsigned long extent_in[6];
    in_mesh->get_extent(extent_in);

    unsigned long extent_out[6];
    memcpy(extent_out, extent_in, 6*sizeof(unsigned long));

    // ensure x-axis is in 0 to 360
    bool shifted_x = false;
    p_teca_variant_array out_x;
    p_teca_unsigned_long_array shift_map;
    if (this->enable_periodic_shift_x &&
        internals::periodic_shift_x(out_x, shift_map, in_x, shifted_x))
    {
        TECA_ERROR("Failed to apply periodic shift to the x-axis")
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
            out_mesh->set_extent(extent_out);

            unsigned long whole_extent[6];
            in_mesh->get_whole_extent(whole_extent);
            whole_extent[1] -= 1;

            out_mesh->set_whole_extent(whole_extent);
        }

        if (internals::periodic_shift_x(out_mesh->get_point_arrays(),
            attributes, shift_map, extent_in, extent_out))
        {
            TECA_ERROR("Failed to apply periodic shift in the x direction")
            return nullptr;
        }
    }

    // ensure y-axis ascending
    bool reordered_y = false;
    p_teca_variant_array out_y;
    if (this->enable_y_axis_ascending &&
        internals::ascending_order_y(out_y, in_y, reordered_y))
    {
        TECA_ERROR("Failed to put the y-axis in ascending order")
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

        std::string var;
        in_mesh->get_y_coordinate_variable(var);
        out_mesh->set_y_coordinates(var, out_y);

        if (internals::ascending_order_y(out_mesh->get_point_arrays(),
            attributes, extent_out))
        {
            TECA_ERROR("Failed to put point arrays into ascending order")
            return nullptr;
        }
    }

    return out_mesh;
}
