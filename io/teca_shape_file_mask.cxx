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

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#include <cuda.h>
#include <cuda_runtime.h>
using stream_vec_t = teca_cuda_util::cuda_stream_vector;
#endif

//#define TECA_DEBUG

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

using poly_coord_t = double;
using polygon_t = teca_geometry::polygon<poly_coord_t>;


/// internal state.
struct teca_shape_file_mask::internals_t
{
    std::vector<polygon_t> polys;   ///< the collection of polygons
                                    ///  defining the region to mask
};

namespace
{
namespace cpu
{
/** search a tile of interest for mesh points that are covered by the
 * given polygon. the mask is assigned 1 where the point is covered by
 * the polygon, otherwise it is not changed.
 *
 * @param[out] mask assigned 1 where the mesh point is covered
 * @param[in] x the mesh's x-coordinates
 * @param[in] y the mesh's y-coordinates
 * @param[in] tile a four-tuple of indices [i0, i1, j0, j1] defining the mesh
 *                 area to search
 * @param[in] nx the mesh size in the x-direction
 * @param[in] poly a polygon defining the region to mask
 *
 * @returns zero if the operation was successful
 */
template <typename mask_t, typename coord_t>
int compute_mask(mask_t *mask, const coord_t *x, const coord_t *y,
    const unsigned long *tile, unsigned long nx, const polygon_t &poly)
{
    for (unsigned long j = tile[2]; j <= tile[3]; ++j)
    {
        for (unsigned long i = tile[0]; i <= tile[1]; ++i)
        {
            if (poly.inside(x[i], y[j]))
                mask[j*nx + i] = 1;
        }
    }
    return 0;
}
}

#if defined(TECA_HAS_CUDA)
namespace cuda_gpu
{
// **************************************************************************
template <typename mask_t, typename coord_t, typename poly_coord_t>
__global__
void compute_mask(mask_t *mask, const coord_t *x, const coord_t *y,
    unsigned long i0, unsigned long j0, unsigned long ni, unsigned long nj,
    unsigned long nx, const poly_coord_t *poly_x, const poly_coord_t *poly_y,
    unsigned long poly_size)
{
    // get the index into the ni by nj tile
    unsigned long q = teca_cuda_util::thread_id_to_array_index();

    unsigned long i = q % ni;
    unsigned long j = q / ni;

    // check for indices outside the tile
    if ((i >= ni) || (j >= nj))
        return;

    // convert from tile to mesh indices
    i += i0;
    j += j0;

    // check if the mesh point is inside the polygon
    if (teca_geometry::point_in_poly<poly_coord_t>(x[i], y[j], poly_x, poly_y, poly_size))
        mask[j*nx + i] = mask_t(1);
}

/** search a tile of interest for mesh points that are covered by the
 * given polygon. the mask is assigned 1 where the point is covered by
 * the polygon, otherwise it is not changed.
 *
 * @param[in] device_id the GPU to use
 * @param[in] strm the CUDA stream to submit work to
 * @param[out] mask assigned 1 where the mesh point is covered
 * @param[in] x the mesh's x-coordinates
 * @param[in] y the mesh's y-coordinates
 * @param[in] tile a four-tuple of indices [i0, i1, j0, j1] defining the mesh
 *                 area to search
 * @param[in] nx the mesh size in the x-direction
 * @param[in] poly a polygon defining the region to mask
 *
 * @returns zero if the operation was successful
 */
template <typename mask_t, typename coord_t>
int compute_mask(int device_id, cudaStream_t strm,
    mask_t *mask, const coord_t *x, const coord_t *y,
    unsigned long *tile, unsigned long nx,
    const polygon_t &poly)
{
    cudaError_t ierr = cudaSuccess;

    // move the polygon vertices to the gpu
    using move_alloc = hamr::cuda_malloc_async_allocator<polygon_t::element_type>;

    auto poly_x = move_alloc::allocate(strm, poly.n_verts, poly.vx.get());
    auto poly_y = move_alloc::allocate(strm, poly.n_verts, poly.vy.get());

    // tile size
    unsigned long ni = tile[1] - tile[0] + 1;
    unsigned long nj = tile[3] - tile[2] + 1;
    unsigned long nij = ni*nj;

    // partition GPU threads over the tile of interest
    dim3 tgrid;
    dim3 bgrid;
    int nblk = 0;
    if (teca_cuda_util::partition_thread_blocks(device_id, nij, 8, bgrid, nblk, tgrid))
    {
        TECA_ERROR("Failed to partition the tile")
        return -1;
    }

    // visit each mesh point and assign 1 if it is covered by the polygon
    compute_mask<<<bgrid,tgrid,0,strm>>>(mask, x, y, tile[0], tile[2],
        ni, nj, nx, poly_x.get(), poly_y.get(), poly.n_verts);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch the compute_mask CUDA kernel"
            << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}
}
#endif
}




// --------------------------------------------------------------------------
teca_shape_file_mask::teca_shape_file_mask() :
    normalize_coordinates(1), number_of_cuda_streams(256),
    internals(new teca_shape_file_mask::internals_t)
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
            "Set this flag to transform the x coordinates of the polygons"
            " from [-180, 180] to [0, 360]")
        TECA_POPTS_GET(int, prefix, number_of_cuda_streams,
            "Set the number of CUDA streams to parallelize the search of mesh"
            " points for intersections with individual polygons in the shapefile")
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
    TECA_POPTS_SET(opts, int, prefix, number_of_cuda_streams)
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

    if (this->internals->polys.empty())
    {
        // load the polygons
        if (teca_shape_file_util::load_polygons(this->get_communicator(),
            this->shape_file, this->internals->polys, this->normalize_coordinates,
            this->verbose))
        {
            TECA_FATAL_ERROR("Failed to read polygons from \"" << this->shape_file << "\"")
            return teca_metadata();
        }
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

    int init = 0;
    int rank = 0;
#if defined(TECA_HAS_MPI)
    MPI_Initialized(&init);
    if (init)
    {
        MPI_Comm_rank(this->get_communicator(), &rank);
    }
#endif
    int verbose = this->get_verbose();

    // catch a potential user error, no polygons
    unsigned long np = this->internals->polys.size();
    if (np < 1)
    {
        TECA_FATAL_ERROR("Failed to compute the mask because no"
            " polygons were loaded")
        return nullptr;
    }

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

    // get the device to run on
    allocator alloc = allocator::malloc;
#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);
    stream_vec_t streams;
    size_t n_streams = 1;
    if (device_id >= 0)
    {
        teca_cuda_util::set_device(device_id);
        alloc = allocator::cuda_async;

        // create the cuda streams. at least one, at most num polygons
        n_streams = std::max(size_t(1), std::min(np,
            (size_t)this->number_of_cuda_streams));

        if (streams.resize(n_streams))
        {
            TECA_FATAL_ERROR("Failed to create " << n_streams << " CUDA streams")
            return nullptr;
        }
    }
#endif

    if (verbose && (rank == 0))
    {
        std::ostringstream oss;
        oss << "Computing shape file mask with " << np << " polygons on ";
#if defined(TECA_HAS_CUDA)
        if (device_id >= 0)
        {
            oss << "GPU " << device_id << " w/ " << n_streams << " streams";
        }
        else
#endif
        {
            oss << "the CPU";
        }
        TECA_STATUS(<< oss.str())
    }

    // allocate the mask array
    unsigned long nx = x->size();
    unsigned long ny = y->size();
    unsigned long nxy = nx*ny;

    auto [mask, p_mask] = ::New<teca_char_array>(nxy, char(0), alloc);

    VARIANT_ARRAY_DISPATCH_FP(x.get(),

        assert_type<CTT>(y);

        // get the coordinate arrays. always needed on the CPU. maybe needed on
        // the GPU if we are assigned to run there.
        auto [sp_x, p_x, sp_y, p_y] = get_host_accessible<CTT>(x, y);

#if defined(TECA_HAS_CUDA)
        CSP dsp_x, dsp_y;
        CNT *dp_x = nullptr, *dp_y = nullptr;

        if (device_id >= 0)
            std::tie(dsp_x, dp_x, dsp_y, dp_y) = get_cuda_accessible<CTT>(x, y);
#endif

        sync_host_access_any(x, y);

        // get the mesh bounds. we'll skip polygons the do not intersect the
        // mesh.
        poly_coord_t mesh_bounds[6] = {p_x[0], p_x[nx-1], p_y[0], p_y[ny-1], 0., 0.};

        // check for ascending coordinates
        if ((mesh_bounds[0] > mesh_bounds[1]) || (mesh_bounds[2] > mesh_bounds[3]))
        {
            TECA_FATAL_ERROR("Ascending mesh coordinates are required")
            return nullptr;
        }

        // visit each polygon
        for (unsigned long p = 0; p < np; ++p)
        {
            // get the polygon's axis aligned bounding box
            poly_coord_t poly_bounds[6] = {0.0};
            this->internals->polys[p].get_bounds(poly_bounds);

            // interset the polygon's AABB with the mesh's AABB.
            poly_coord_t int_bounds[6] = {0.0};
            teca_coordinate_util::intersect_tiles(int_bounds, mesh_bounds, poly_bounds);

            // check for empty intersection. in that case we can skip the
            // search of the mesh
            if (teca_coordinate_util::empty_tile(int_bounds))
            {
                if ((verbose > 1) && (rank == 0))
                    std::cerr << "skipped polygon " << p << " with bounds "
                        << poly_bounds << " outside mesh bounds " << mesh_bounds
                        << std::endl;
                continue;
            }

            // convert the overlapping region to the mesh extents. this narrows
            // the search space to only mesh points that could intersect.
            unsigned long int_extent[6] = {0};
            if (teca_coordinate_util::bounds_to_extent(int_bounds  , p_x, nx, int_extent  ) ||
                teca_coordinate_util::bounds_to_extent(int_bounds+2, p_y, ny, int_extent+2))
            {
                TECA_FATAL_ERROR("Failed to convert the intersection of polygon "
                    << p << " and the mesh, bounds [" << int_bounds[0] << ", "
                    << int_bounds[1] << ", " << int_bounds[2] << ", "
                    << int_bounds[3] << " to a valid mesh extent")
                continue;
            }

            // test each mesh point in the overlapping region for intersection
            // with the polygon
#if defined(TECA_HAS_CUDA)
            if (device_id >= 0)
            {
                // use a stream to overlap this iteration with the others
                cudaStream_t strm = streams[ p % n_streams ];

                if (::cuda_gpu::compute_mask(device_id, strm, p_mask, dp_x, dp_y,
                    int_extent, nx, this->internals->polys[p]))
                {
                    TECA_ERROR("Failed to compute the mask using CUDA")
                    return nullptr;
                }
            }
            else
            {
#endif
                if (::cpu::compute_mask(p_mask, p_x, p_y,
                    int_extent, nx, this->internals->polys[p]))
                {
                    TECA_ERROR("Failed to compute the mask using the CPU")
                    return nullptr;
                }
#if defined(TECA_HAS_CUDA)
            }
#endif
        }
        )

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
