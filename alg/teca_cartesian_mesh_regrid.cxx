#include "teca_cartesian_mesh_regrid.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh_util.h"

#include <algorithm>
#include <iostream>

using std::string;
using std::vector;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// convert an extent from the target coordinate system
// into an extent in the source coordinate system
// get the extent in the source
int convert_extent(
    const vector<unsigned long> &target_ext,
    const const_p_teca_variant_array &target_x,
    const const_p_teca_variant_array &target_y,
    const const_p_teca_variant_array &target_z,
    const const_p_teca_variant_array &source_x,
    const const_p_teca_variant_array &source_y,
    const const_p_teca_variant_array &source_z,
    vector<unsigned long> &source_ext)
{
    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        target_x.get(),
        1,

        const NT1 *p_target_x = std::dynamic_pointer_cast<TT1>(target_x)->get();
        const NT1 *p_target_y = std::dynamic_pointer_cast<TT1>(target_y)->get();
        const NT1 *p_target_z = std::dynamic_pointer_cast<TT1>(target_z)->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            const teca_variant_array_impl,
            source_x.get(),
            2,

            const NT2 *p_source_x = std::dynamic_pointer_cast<TT2>(source_x)->get();
            const NT2 *p_source_y = std::dynamic_pointer_cast<TT2>(source_y)->get();
            const NT2 *p_source_z = std::dynamic_pointer_cast<TT2>(source_z)->get();

            return bounds_to_extent(
                static_cast<NT2>(p_target_x[target_ext[0]]),
                static_cast<NT2>(p_target_x[target_ext[1]]),
                static_cast<NT2>(p_target_y[target_ext[2]]),
                static_cast<NT2>(p_target_y[target_ext[3]]),
                static_cast<NT2>(p_target_z[target_ext[4]]),
                static_cast<NT2>(p_target_z[target_ext[5]]),
                p_source_x, p_source_y, p_source_z,
                source_x->size() - 1,
                source_y->size() - 1,
                source_z->size() - 1,
                true, source_ext);
            )
        )

    // unsupported coordinate type
    return -1;
}

// 0 order (nearest neighbor) interpolation
// for nodal data on stretched cartesian mesh.
template<typename CT, typename DT>
int interpolate_nearest(
    CT cx,
    CT cy,
    CT cz,
    const CT *p_x,
    const CT *p_y,
    const CT *p_z,
    const DT *p_data,
    unsigned long ihi,
    unsigned long jhi,
    unsigned long khi,
    unsigned long nx,
    unsigned long nxy,
    DT &val)
{
    // get i,j of node less than cx,cy
    unsigned long i = 0;
    unsigned long j = 0;
    unsigned long k = 0;

    if ((ihi && index_of(p_x, 0, ihi, cx, true, i))
        || (jhi && index_of(p_y, 0, jhi, cy, true, j))
        || (khi && index_of(p_z, 0, khi, cz, true, k)))
    {
        // cx,cy,cz is outside the coordinate axes
        return -1;
    }

    // get i,j of node greater than cx,cy
    unsigned long ii = std::min(i + 1, ihi);
    unsigned long jj = std::min(j + 1, jhi);
    unsigned long kk = std::min(k + 1, khi);

    // get index of nearest node
    unsigned long p = (cx - p_x[i]) <= (p_x[ii] - cx) ? i : ii;
    unsigned long q = (cy - p_y[j]) <= (p_y[jj] - cy) ? j : jj;
    unsigned long r = (cz - p_z[k]) <= (p_z[kk] - cz) ? k : kk;

    // assign value from nearest node
    val = p_data[p + nx*q + nxy*r];
    return 0;
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_regrid::teca_cartesian_mesh_regrid()
    : interpolation_mode(nearest)
{
    this->set_number_of_input_connections(2);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_regrid::~teca_cartesian_mesh_regrid()
{}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_regrid::clear_arrays()
{
    this->source_arrays.clear();
    this->target_arrays.clear();
    this->set_modified();
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_regrid::add_array(const std::string &array)
{
    this->add_array(array, "");
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_regrid::add_array(
    const std::string &source_array,
    const std::string &target_array)
{
    this->source_arrays.push_back(source_array);
    this->target_arrays.push_back(target_array);
    this->set_modified();
}

// --------------------------------------------------------------------------
teca_metadata teca_cartesian_mesh_regrid::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_reader::get_output_metadata" << endl;
#endif
    (void)port;

    // copy metadata from the target input then
    // append variable metadata from the source input
    teca_metadata output_md(input_md[0]);
    output_md.append("variables", input_md[1].get("variables"));
    output_md.append("attributes", input_md[1].get("attributes"));
    output_md.append("time_vars", input_md[1].get("time_vars"));

    return output_md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_cartesian_mesh_regrid::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;

    vector<teca_metadata> up_reqs(2);

    // compose the target request
    teca_metadata target_req(request);

    // clear our keys
    target_req.remove("regrid_source_arrays");
    target_req.remove("regrid_target_arrays");

    // compose the source request
    // merge in named arrays
    vector<string> arrays;
    request.get("regrid_source_arrays", arrays);

    std::copy(this->source_arrays.begin(),
        this->source_arrays.end(), std::back_inserter(arrays));

    // TODO -- check that each array has attribute centering=point

    // get the target extent and coordinates
    vector<unsigned long> target_extent(6, 0l);
    request.get("extent", target_extent);

    teca_metadata target_coords;
    p_teca_variant_array target_x;
    p_teca_variant_array target_y;
    p_teca_variant_array target_z;

    if (input_md[0].get("coordinates", target_coords)
        || !(target_x = target_coords.get("x"))
        || !(target_y = target_coords.get("y"))
        || !(target_z = target_coords.get("z")))
    {
        TECA_ERROR("failed to locate target mesh coordinates")
        return up_reqs;
    }

    // get the source extent and coordinates
    teca_metadata source_coords;
    p_teca_variant_array source_x;
    p_teca_variant_array source_y;
    p_teca_variant_array source_z;

    if (input_md[1].get("coordinates", source_coords)
        || !(source_x = source_coords.get("x"))
        || !(source_y = source_coords.get("y"))
        || !(source_z = source_coords.get("z")))
    {
        TECA_ERROR("failed to locate source mesh coordinates")
        return up_reqs;
    }

    // map the target extent to the source mesh
    vector<unsigned long> source_extent(6, 0l);
    if (convert_extent(
        target_extent,
        target_x, target_y, target_z,
        source_x, source_y, source_z,
        source_extent))
    {
        TECA_ERROR("failed to convert from target to source extent")
        return up_reqs;
    }

    // build the request
    teca_metadata source_req(target_req);
    source_req.insert("arrays", arrays);
    source_req.insert("extent", source_extent);

    // TODO -- how should time be handled? we could
    // interpolate data from source input to the time value
    // requested. but this should be handled upstream
    // in a dedicated filter.

    // send the requests up
    up_reqs[0] = target_req;
    up_reqs[1] = source_req;

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_regrid::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cartesian_mesh_regrid::execute" << endl;
#endif
    (void)port;

    p_teca_cartesian_mesh in_target
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(input_data[0]));

    const_p_teca_cartesian_mesh source
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[1]);

    if (!in_target || !source)
    {
        TECA_ERROR("invalid input. target invalid "
            << !in_target << " source invalid " << !source )
        return nullptr;
    }

    // create the output
    p_teca_cartesian_mesh target = teca_cartesian_mesh::New();
    target->shallow_copy(in_target);

    // get the source and targent extents
    vector<unsigned long> source_ext;
    source->get_extent(source_ext);

    vector<unsigned long> target_ext;
    target->get_extent(target_ext);

    // get the list of arrays to move
    vector<string> source_arrays;
    request.get("regrid_source_source_arrays", source_arrays);
    std::copy(this->source_arrays.begin(),
        this->source_arrays.end(), std::back_inserter(source_arrays));

    vector<string> target_arrays;
    request.get("regrid_source_target_arrays", target_arrays);
    std::copy(this->target_arrays.begin(),
        this->target_arrays.end(), std::back_inserter(target_arrays));

    // verify we have both source and target array names
    if (source_arrays.size() != target_arrays.size())
    {
        TECA_ERROR("source and target array names mismatch")
        return nullptr;
    }

    // move the arrays
    const_p_teca_variant_array target_xc = target->get_x_coordinates();
    const_p_teca_variant_array target_yc = target->get_y_coordinates();
    const_p_teca_variant_array target_zc = target->get_z_coordinates();
    p_teca_array_collection target_ac = target->get_point_arrays();

    unsigned long target_nx = target_xc->size();
    unsigned long target_ny = target_yc->size();
    unsigned long target_nz = target_zc->size();
    unsigned long target_size = target_nx*target_ny*target_nz;

    const_p_teca_variant_array source_xc = source->get_x_coordinates();
    const_p_teca_variant_array source_yc = source->get_y_coordinates();
    const_p_teca_variant_array source_zc = source->get_z_coordinates();
    const_p_teca_array_collection source_ac = source->get_point_arrays();

    unsigned long source_nx = source_xc->size();
    unsigned long source_ny = source_yc->size();
    unsigned long source_nz = source_zc->size();
    unsigned long source_nxy = source_nx*source_ny;
    unsigned long source_ihi = source_nx - 1;
    unsigned long source_jhi = source_ny - 1;
    unsigned long source_khi = source_nz - 1;

    NESTED_TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        target_xc.get(),
        1,

        const NT1 *p_target_xc = std::dynamic_pointer_cast<TT1>(target_xc)->get();
        const NT1 *p_target_yc = std::dynamic_pointer_cast<TT1>(target_yc)->get();
        const NT1 *p_target_zc = std::dynamic_pointer_cast<TT1>(target_zc)->get();

        NESTED_TEMPLATE_DISPATCH_FP(
            const teca_variant_array_impl,
            source_xc.get(),
            2,

            const NT2 *p_source_xc = std::dynamic_pointer_cast<TT2>(source_xc)->get();
            const NT2 *p_source_yc = std::dynamic_pointer_cast<TT2>(source_yc)->get();
            const NT2 *p_source_zc = std::dynamic_pointer_cast<TT2>(source_zc)->get();

            size_t n_arrays = source_arrays.size();
            for (size_t i = 0; i < n_arrays; ++i)
            {
                const_p_teca_variant_array source_a = source_ac->get(source_arrays[i]);
                p_teca_variant_array target_a = source_a->new_instance();
                target_a->resize(target_size);

                NESTED_TEMPLATE_DISPATCH(
                    teca_variant_array_impl,
                    target_a.get(),
                    3,

                    const NT3 *p_source_a = std::dynamic_pointer_cast<const TT3>(source_a)->get();
                    NT3 *p_target_a = std::dynamic_pointer_cast<TT3>(target_a)->get();

                    unsigned long q = 0;
                    for (unsigned long k = 0; k < target_nz; ++k)
                    {
                        for (unsigned long j = 0; j < target_ny; ++j)
                        {
                            for (unsigned long i = 0; i < target_nx; ++i, ++q)
                            {
                                if (interpolate_nearest<NT2,NT3>(
                                    static_cast<NT2>(p_target_xc[i]),
                                    static_cast<NT2>(p_target_yc[j]),
                                    static_cast<NT2>(p_target_zc[k]),
                                    p_source_xc, p_source_yc, p_source_zc,
                                    p_source_a, source_ihi, source_jhi, source_khi,
                                    source_nx, source_nxy,
                                    p_target_a[q]))
                                {
                                    TECA_ERROR("failed to interpolate " << i << ", " << j << ", " << k)
                                    return nullptr;
                                }
                            }
                        }
                    }

                    const string &name
                        = target_arrays[i].size() ? target_arrays[i] : source_arrays[i];

                    target_ac->set(name, target_a);
                    )
                }
                )
            )

    return target;
}
