#include "teca_cartesian_mesh_regrid.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

using std::cerr;
using std::endl;

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

//#define TECA_DEBUG

// always use nearest neighbor interpolation for integers
// to avoid truncation errors. an alternative would be to
// implement rounding in the interpolator for integer types
template <typename data_t>
int get_interpolation_mode(int desired_mode,
    typename std::enable_if<std::is_integral<data_t>::value>::type* = 0)
{
    (void)desired_mode;
    return teca_cartesian_mesh_regrid::nearest;
}

// use the requested interpolation mode for floating point
// data
template <typename data_t>
int get_interpolation_mode(int desired_mode,
    typename std::enable_if<std::is_floating_point<data_t>::value>::type* = 0)
{
    return desired_mode;
}


// 3D
template<typename NT1, typename NT2, typename NT3, class interp_t>
int interpolate(unsigned long target_nx, unsigned long target_ny,
    unsigned long target_nz, const NT1 *p_target_xc, const NT1 *p_target_yc,
    const NT1 *p_target_zc, NT3 *p_target_a, const NT2 *p_source_xc,
    const NT2 *p_source_yc, const NT2 *p_source_zc, const NT3 *p_source_a,
    unsigned long source_ihi, unsigned long source_jhi, unsigned long source_khi,
    unsigned long source_nx, unsigned long source_nxy)
{
    interp_t f;
    unsigned long q = 0;
    for (unsigned long k = 0; k < target_nz; ++k)
    {
        NT2 tz = static_cast<NT2>(p_target_zc[k]);
        for (unsigned long j = 0; j < target_ny; ++j)
        {
            NT2 ty = static_cast<NT2>(p_target_yc[j]);
            for (unsigned long i = 0; i < target_nx; ++i, ++q)
            {
                NT2 tx = static_cast<NT2>(p_target_xc[i]);
                if (f(tx,ty,tz,
                    p_source_xc, p_source_yc, p_source_zc,
                    p_source_a, source_ihi, source_jhi, source_khi,
                    source_nx, source_nxy,
                    p_target_a[q]))
                {
                    TECA_ERROR("failed to interpolate i=(" << i << ", " << j << ", " << k
                        << ") x=(" << tx << ", " << ty << ", " << tz << ")")
                    return -1;
                }
            }
        }
    }
    return 0;
}

// 2D - x-y
template<typename NT1, typename NT2, typename NT3, class interp_t>
int interpolate(unsigned long target_nx, unsigned long target_ny,
    const NT1 *p_target_xc, const NT1 *p_target_yc,
    NT3 *p_target_a, const NT2 *p_source_xc,
    const NT2 *p_source_yc, const NT3 *p_source_a,
    unsigned long source_ihi, unsigned long source_jhi,
    unsigned long source_nx)
{
    interp_t f;
    unsigned long q = 0;
    for (unsigned long j = 0; j < target_ny; ++j)
    {
        NT2 ty = static_cast<NT2>(p_target_yc[j]);
        for (unsigned long i = 0; i < target_nx; ++i, ++q)
        {
            NT2 tx = static_cast<NT2>(p_target_xc[i]);
            if (f(tx,ty,
                p_source_xc, p_source_yc,
                p_source_a, source_ihi, source_jhi,
                source_nx, p_target_a[q]))
            {
                TECA_ERROR("failed to interpolate i=(" << i << ", " << j
                    << ") x=(" << tx << ", " << ty << ", " << ")")
                return -1;
            }
        }
    }
    return 0;
}

template<typename taget_coord_t, typename source_coord_t, typename array_t>
int interpolate(int mode, unsigned long target_nx, unsigned long target_ny,
    unsigned long target_nz, const taget_coord_t *p_target_xc,
    const taget_coord_t *p_target_yc, const taget_coord_t *p_target_zc,
    array_t *p_target_a, const source_coord_t *p_source_xc,
    const source_coord_t *p_source_yc, const source_coord_t *p_source_zc,
    const array_t *p_source_a, unsigned long source_ihi, unsigned long source_jhi,
    unsigned long source_khi, unsigned long source_nx, unsigned long source_ny,
    unsigned long source_nz)
{
    using nearest_interp_t = teca_coordinate_util::interpolate_t<0>;
    using linear_interp_t = teca_coordinate_util::interpolate_t<1>;

    unsigned long source_nxy = source_nx*source_ny;

    switch (get_interpolation_mode<array_t>(mode))
    {
        case teca_cartesian_mesh_regrid::nearest:
        {
            if ((target_nz == 1) && (source_nz == 1))
            {
                // 2D in the x-y plane
                return interpolate<taget_coord_t,
                    source_coord_t, array_t, nearest_interp_t>(
                        target_nx, target_ny, p_target_xc, p_target_yc,
                        p_target_a, p_source_xc, p_source_yc, p_source_a,
                        source_ihi, source_jhi, source_nx);
            }
            else
            {
                // 3D
                return interpolate<taget_coord_t,
                    source_coord_t, array_t, nearest_interp_t>(
                        target_nx, target_ny, target_nz, p_target_xc,
                        p_target_yc, p_target_zc, p_target_a, p_source_xc,
                        p_source_yc, p_source_zc, p_source_a, source_ihi,
                        source_jhi, source_khi, source_nx, source_nxy);
            }
            break;
        }
        case teca_cartesian_mesh_regrid::linear:
        {
            if ((target_nz == 1) && (source_nz == 1))
            {
                // 2D in the x-y plane
                return interpolate<taget_coord_t,
                    source_coord_t, array_t, linear_interp_t>(
                        target_nx, target_ny, p_target_xc, p_target_yc,
                        p_target_a, p_source_xc, p_source_yc, p_source_a,
                        source_ihi, source_jhi, source_nx);
            }
            else
            {
                // 3D
                return interpolate<taget_coord_t,
                    source_coord_t, array_t, linear_interp_t>(
                        target_nx, target_ny, target_nz, p_target_xc,
                        p_target_yc, p_target_zc, p_target_a, p_source_xc,
                        p_source_yc, p_source_zc, p_source_a, source_ihi,
                        source_jhi, source_khi, source_nx, source_nxy);
            }
            break;
        }
    }

    TECA_ERROR("invalid interpolation mode \"" << mode << "\"")
    return -1;
}




// --------------------------------------------------------------------------
teca_cartesian_mesh_regrid::teca_cartesian_mesh_regrid()
    : target_input(0), interpolation_mode(nearest)
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_regrid::~teca_cartesian_mesh_regrid()
{}

#if defined(TECA_HAS_BOOST)

// --------------------------------------------------------------------------
void teca_cartesian_mesh_regrid::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_regrid":prefix));

    opts.add_options()
        TECA_POPTS_GET(int, prefix, target_input,
            "select input connection that contains metadata")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, arrays,
            "list of arrays to move from source to target mesh")
        TECA_POPTS_GET(int, prefix, interpolation_mode,
            "linear or nearest interpolation")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_regrid::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, int, prefix, target_input)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, arrays)
    TECA_POPTS_SET(opts, int, prefix, interpolation_mode)
}

#endif

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_regrid::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_regrid::get_output_metadata" << endl;
#endif
    (void)port;

    int md_src = this->target_input ? 0 : 1;
    int md_tgt = this->target_input ? 1 : 0;

    // start with a copy of metadata from the target
    teca_metadata output_md(input_md[md_tgt]);

    // get target metadata
    std::vector<std::string> target_vars;
    input_md[md_tgt].get("variables", target_vars);

    teca_metadata target_atts;
    input_md[md_tgt].get("attributes", target_atts);

    // get source metadata
    std::vector<std::string> source_vars;
    input_md[md_src].get("variables", source_vars);

    teca_metadata source_atts;
    input_md[md_src].get("attributes", source_atts);

    // merge metadata from source and target
    // variables should be unique lists.
    // attributes are indexed by variable names
    // in the case of collisions, the target variable
    // is kept, the source variable is ignored
    size_t n_source_vars = source_vars.size();
    for (size_t i = 0; i < n_source_vars; ++i)
    {
        const std::string &source = source_vars[i];

        // check that there's not a variable of that same name in target
        std::vector<std::string>::iterator first = target_vars.begin();
        std::vector<std::string>::iterator last = target_vars.end();
        if (find(first, last, source) == last)
        {
            // not present in target, ok to add it
            target_vars.push_back(source);

            teca_metadata atts;
            source_atts.get(source, atts);
            target_atts.set(source, atts);
        }

    }

    // update with merged lists
    output_md.set("variables", target_vars);
    output_md.set("attributes", target_atts);

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cartesian_mesh_regrid::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    std::vector<teca_metadata> up_reqs(2);

    // route requests for arrays to either target or the source.
    // if the array exists in both then it is take from the target
    teca_metadata target_req(request);
    teca_metadata source_req(request);

    // get target metadata
    int md_tgt = this->target_input ? 1 : 0;
    std::vector<std::string> target_vars;
    input_md[md_tgt].get("variables", target_vars);
    std::vector<std::string>::iterator tgt_0 = target_vars.begin();
    std::vector<std::string>::iterator tgt_1 = target_vars.end();

    // get source metadata
    int md_src = this->target_input ? 0 : 1;
    std::vector<std::string> source_vars;
    input_md[md_src].get("variables", source_vars);
    std::vector<std::string>::iterator src_0 = source_vars.begin();
    std::vector<std::string>::iterator src_1 = source_vars.end();

    // get the requested arrays
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    // add any explicitly named
    std::copy(this->arrays.begin(), this->arrays.end(),
        std::back_inserter(req_arrays));

    // route. if found in the target then  routed there, else
    // if found in the source routed there, else skipped
    std::vector<std::string> target_arrays;
    std::vector<std::string> source_arrays;

    std::vector<std::string>::iterator it = req_arrays.begin();
    std::vector<std::string>::iterator end = req_arrays.end();
    for (; it != end; ++it)
    {
        if (find(tgt_0, tgt_1, *it) != tgt_1)
            target_arrays.push_back(*it);
        else if (find(src_0, src_1, *it) != src_1)
            source_arrays.push_back(*it);
        else
            TECA_FATAL_ERROR("\"" << *it << "\" was not in target nor source")
    }

    target_req.set("arrays", target_arrays);
    source_req.set("arrays", source_arrays);

    // get the target extent and coordinates
    teca_metadata target_coords;
    p_teca_variant_array target_x, target_y, target_z;

    if (input_md[md_tgt].get("coordinates", target_coords)
        || !(target_x = target_coords.get("x"))
        || !(target_y = target_coords.get("y"))
        || !(target_z = target_coords.get("z")))
    {
        TECA_FATAL_ERROR("failed to locate target mesh coordinates")
        return up_reqs;
    }

    // get the actual bounds of what we will be served with. this will be a
    // region covering the requested bounds. we need to insure that source data
    // covers this region, not just the requested bounds.
    double request_bounds[6] = {0.0};
    unsigned long target_extent[6] = {0l};
    if (request.get("bounds", request_bounds, 6))
    {
        if (request.get("extent", target_extent, 6))
        {
            target_extent[1] = target_x->size() - 1;
            target_extent[3] = target_y->size() - 1;
            target_extent[5] = target_z->size() - 1;
        }
    }
    else
    {
        if (teca_coordinate_util::bounds_to_extent(request_bounds,
                target_x, target_y, target_z, target_extent) ||
            teca_coordinate_util::validate_extent(target_x->size(),
                target_y->size(), target_z->size(), target_extent, true))
        {
            TECA_FATAL_ERROR("invalid bounds requested [" << request_bounds[0]  << ", "
                << request_bounds[1] << ", " << request_bounds[2] << ", "
                << request_bounds[3] << ", " << request_bounds[4] << ", "
                << request_bounds[5] << "]")
            return up_reqs;
        }
    }

    double target_bounds[6] = {0.0};
    target_x->get(target_extent[0], target_bounds[0]);
    target_x->get(target_extent[1], target_bounds[1]);
    target_y->get(target_extent[2], target_bounds[2]);
    target_y->get(target_extent[3], target_bounds[3]);
    target_z->get(target_extent[4], target_bounds[4]);
    target_z->get(target_extent[5], target_bounds[5]);

    // if the source is 2D, the cf_reader may have faked the vertical dimension.
    // in that case, use the source's vertical coordinate in the requested bounds
    teca_metadata source_coords;
    p_teca_variant_array source_z;

    if (input_md[md_src].get("coordinates", source_coords)
        || !(source_z = source_coords.get("z")))
    {
        TECA_FATAL_ERROR("failed to locate source mesh coordinates")
        return up_reqs;
    }

    if (source_z->size() == 1)
    {
        source_z->get(0, target_bounds[4]);
        source_z->get(0, target_bounds[5]);
    }

    // send the target bounds to the source as well
    source_req.set("bounds", target_bounds, 6);

    // send the requests up
    up_reqs[md_tgt] = target_req;
    up_reqs[md_src] = source_req;

#ifdef TECA_DEBUG
    std::cerr << "source request = ";
    source_req.to_stream(std::cerr);
    std::cerr << std::endl;

    std::cerr << "target request = ";
    target_req.to_stream(std::cerr);
    std::cerr << std::endl;
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_regrid::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_regrid::execute" << endl;
#endif
    (void)port;

    int md_src = this->target_input ? 0 : 1;
    int md_tgt = this->target_input ? 1 : 0;

    p_teca_cartesian_mesh in_target
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(input_data[md_tgt]));

    const_p_teca_cartesian_mesh source
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[md_src]);

    if (!in_target || !source)
    {
        TECA_FATAL_ERROR("invalid input. target invalid "
            << !in_target << " source invalid " << !source )
        return nullptr;
    }

    // create the output
    p_teca_cartesian_mesh target = teca_cartesian_mesh::New();
    target->shallow_copy(in_target);

    // get the source and targent extents
    std::vector<unsigned long> source_ext;
    source->get_extent(source_ext);

    std::vector<unsigned long> target_ext;
    target->get_extent(target_ext);

    // get the list of arrays to move
    std::vector<std::string> req_arrays;
    request.get("arrays", req_arrays);

    // add any explicitly named
    std::copy(this->arrays.begin(), this->arrays.end(),
        std::back_inserter(req_arrays));

    // route. if found in the target then  routed there, else
    // if found in the source routed there, else skipped
    std::vector<std::string> source_arrays;

    std::vector<std::string>::iterator it = req_arrays.begin();
    std::vector<std::string>::iterator end = req_arrays.end();
    for (; it != end; ++it)
    {
        if (!target->get_point_arrays()->has(*it))
        {
            if (source->get_point_arrays()->has(*it))
            {
                source_arrays.push_back(*it);
            }
            else
            {
                TECA_FATAL_ERROR("Array \"" << *it
                    << "\" is neither present in source or target mesh")
                return nullptr;
            }
        }
    }

    // catch a user error
    if (!source_arrays.size() &&
        teca_mpi_util::mpi_rank_0(this->get_communicator()))
    {
        TECA_WARNING("No arrays will be interpolated")
    }

    // get the coordinate and data arrays
    const_p_teca_variant_array target_xc = target->get_x_coordinates();
    const_p_teca_variant_array target_yc = target->get_y_coordinates();
    const_p_teca_variant_array target_zc = target->get_z_coordinates();
    p_teca_array_collection target_ac = target->get_point_arrays();

    const_p_teca_variant_array source_xc = source->get_x_coordinates();
    const_p_teca_variant_array source_yc = source->get_y_coordinates();
    const_p_teca_variant_array source_zc = source->get_z_coordinates();
    const_p_teca_array_collection source_ac = source->get_point_arrays();

    // move the array attributes
    teca_metadata source_atts;
    source->get_attributes(source_atts);

    teca_metadata target_atts;
    target->get_attributes(target_atts);

    size_t n_arrays = source_arrays.size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        teca_metadata array_atts;
        source_atts.get(source_arrays[i], array_atts);
        target_atts.set(source_arrays[i], array_atts);
    }

    target->set_attributes(target_atts);

    // move the arrays
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // use the array's active dimensions to compute the output array size and
        // interpolation loop bounds
        unsigned long target_shape[4] = {0};
        target->get_array_shape(source_arrays[i], target_shape);

        unsigned long target_nx = target_shape[0];
        unsigned long target_ny = target_shape[1];
        unsigned long target_nz = target_shape[2];

        unsigned long target_size = target_nx*target_ny*target_nz;

        unsigned long target_ihi = target_nx - 1;
        unsigned long target_jhi = target_ny - 1;
        unsigned long target_khi = target_nz - 1;

        unsigned long source_shape[4] = {0};
        source->get_array_shape(source_arrays[i], source_shape);

        unsigned long source_nx = source_shape[0];
        unsigned long source_ny = source_shape[1];
        unsigned long source_nz = source_shape[2];
        unsigned long source_ihi = source_nx - 1;
        unsigned long source_jhi = source_ny - 1;
        unsigned long source_khi = source_nz - 1;

        // copy when the input and output meshes are the same. the meshes are the
        // same when they span the same world space and have the same resolution
        double tx0, tx1, sx0, sx1;
        source_xc->get(0, sx0);
        target_xc->get(0, tx0);
        source_xc->get(source_ihi, sx1);
        target_xc->get(target_ihi, tx1);

        double ty0, ty1, sy0, sy1;
        source_yc->get(0, sy0);
        target_yc->get(0, ty0);
        source_yc->get(source_jhi, sy1);
        target_yc->get(target_jhi, ty1);

        double tz0, tz1, sz0, sz1;
        source_zc->get(0, sz0);
        target_zc->get(0, tz0);
        source_zc->get(source_khi, sz1);
        target_zc->get(target_khi, tz1);

        if ((source_nx == target_nx) && (source_ny == target_ny) && (source_nz == target_nz) &&
            teca_coordinate_util::equal(sx0, tx0) && teca_coordinate_util::equal(sx1, tx1) &&
            teca_coordinate_util::equal(sy0, ty0) && teca_coordinate_util::equal(sy1, ty1) &&
            teca_coordinate_util::equal(sz0, tz0) && teca_coordinate_util::equal(sz1, tz1))
        {
            if (this->verbose)
            {
                TECA_STATUS("Identical dimensions and bounds detected \""
                    << source_arrays[i] << "\". Copying data.")
            }

            const_p_teca_variant_array source_a = source_ac->get(source_arrays[i]);
            p_teca_variant_array target_a = source_a->new_copy();
            target_ac->set(source_arrays[i], target_a);
        }
        else
        {
            if (this->verbose)
            {
                TECA_STATUS("Interpolating data from \"" << source_arrays[i] << "\"")
            }

            NESTED_VARIANT_ARRAY_DISPATCH_FP(
                target_xc.get(), _TGT,

                // get the target cooridnates
                assert_type<CTT_TGT>(target_yc, target_zc);
                auto [sp_target_xc, p_target_xc] = get_host_accessible<CTT_TGT>(target_xc);
                auto [sp_target_yc, p_target_yc] = get_host_accessible<CTT_TGT>(target_yc);
                auto [sp_target_zc, p_target_zc] = get_host_accessible<CTT_TGT>(target_zc);

                NESTED_VARIANT_ARRAY_DISPATCH_FP(
                    source_xc.get(), _SRC,

                    // get the target cooridnates
                    assert_type<CTT_SRC>(source_yc, source_zc);
                    auto [sp_source_xc, p_source_xc] = get_host_accessible<CTT_SRC>(source_xc);
                    auto [sp_source_yc, p_source_yc] = get_host_accessible<CTT_SRC>(source_yc);
                    auto [sp_source_zc, p_source_zc] = get_host_accessible<CTT_SRC>(source_zc);

                    // get the source array
                    const_p_teca_variant_array source_a = source_ac->get(source_arrays[i]);

                    // allocate the target array
                    p_teca_variant_array target_a = source_a->new_instance(target_size);

                    NESTED_VARIANT_ARRAY_DISPATCH(
                        target_a.get(), _DATA,

                        // interpolate
                        auto [sp_source_a, p_source_a] = get_host_accessible<CTT_DATA>(source_a);
                        auto [p_target_a] = data<TT_DATA>(target_a);

                        sync_host_access_any(target_xc, target_yc, target_zc,
                                             source_xc, source_yc, source_zc,
                                             source_a);

                        if (interpolate(this->interpolation_mode, target_nx, target_ny, target_nz,
                            p_target_xc, p_target_yc, p_target_zc, p_target_a, p_source_xc,
                            p_source_yc, p_source_zc, p_source_a, source_ihi, source_jhi,
                            source_khi, source_nx, source_ny, source_nz))
                        {
                            TECA_FATAL_ERROR("Failed to move \"" << source_arrays[i] << "\"")
                            return nullptr;
                        }
                        )
                    else
                    {
                        TECA_FATAL_ERROR("Unsupported array type " << source_a->get_class_name())
                    }

                    target_ac->set(source_arrays[i], target_a);
                    )
                else
                {
                    TECA_FATAL_ERROR("Unupported coordinate type " << source_xc->get_class_name())
                }
                )
            else
            {
                TECA_FATAL_ERROR("Unupported coordinate type " << target_xc->get_class_name())
            }
        }
    }

    return target;
}
