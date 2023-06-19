#include "teca_connected_components.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_attributes.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <cmath>
#include <sstream>

//#define TECA_DEBUG

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

#if defined(TECA_HAS_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>
#include "hamr_buffer.h"

namespace cuda_impl
{

// mask - mask with bits set where input is 1
// tx - cuda thread id
__device__
int start_distance(int mask, int tx)
{
    return __clz(~(mask << (32 - tx)));
}


// mask - mask with bits set where input is 1
// tx - cuda thread id
__device__
int end_distance(int mask, int tx)
{
    return __ffs(~(mask >> (tx + 1)));
}

template <typename T>
__device__
void swap(T &a, T &b)
{
    T c = a;
    a = b;
    b = c;
}

/** this works by finding
the root of the two equivalence trees the labels are belonging
to and writing the minimum root index to the root with the
maximum index.*/
__device__
void merge(int *labels, int l1, int l2)
{
    while ((l1 != l2) && (l1 != labels[l1]))
        l1 = labels[l1];

    while ((l1 != l2) && (l2 != labels[l2]))
        l2 = labels[l2];

    while (l1 != l2)
    {
        if (l1 < l2)
            swap(l1, l2);

        int l3 = atomicMin(&labels[l1], l2);

        l1 = ( l1 == l3 ? l2 : l3 );
    }
}

// the CUDA warp size
#define NUM_THREADS_X 32

// each CUDA block has a horizontal strip this many pixels tall
#define STRIP_HEIGHT 4

// **************************************************************************
template <typename image_t>
__global__
void label_strip(const image_t *image, int *labels, int nx, int ny, bool periodic)
{
    __shared__ int shared_pix[STRIP_HEIGHT];

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    int line_base = y * nx + threadIdx.x;

    int dy = 0;
    int dy1 = 0;

    int maxI = nx % 32 ? (nx / 32 + 1) * 32 : nx;
    for (int i = 0; i < maxI; i += 32)
    {
        bool threadActive = ((x + i) < nx) && (y < ny);
        unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

        int k_yx = line_base + i;
        image_t p_yx = threadActive ? image[k_yx] : 0;
        int pix_y = __ballot_sync(activeMask, p_yx);
        int s_dy = start_distance(pix_y, threadIdx.x);

        if (threadActive && p_yx && (s_dy == 0))
            labels[k_yx] = threadIdx.x == 0 ? k_yx - dy : k_yx;

        if (threadActive && (threadIdx.x == 0))
            shared_pix[threadIdx.y] = pix_y;

        __syncthreads();

        int pix_y1 = !threadActive || (threadIdx.y == 0) ? 0 : shared_pix[threadIdx.y - 1];
        int p_y1x = pix_y1 & (1 << threadIdx.x);
        int s_dy1 = start_distance(pix_y1, threadIdx.x);

        if (threadIdx.x == 0)
        {
            s_dy = dy;
            s_dy1 = dy1;
        }

        if (p_yx && p_y1x && ((s_dy == 0) || (s_dy1 == 0)))
        {
            int label_1 = k_yx - s_dy;
            int label_2 = k_yx - nx - s_dy1;
            merge(labels, label_1, label_2);
        }

        int d = start_distance(pix_y1, 32);
        dy1 = d == 32 ? d + dy1 : d;

        d = start_distance(pix_y, 32);
        dy = d == 32 ? d + dy : d;
    }

    if (periodic)
    {
        bool threadActive = (threadIdx.x == 0) &&  (x < nx) && (y < ny);
        int nx1 = nx - 1;
        if (threadActive && image[line_base] && image[line_base + nx1])
        {
            int pix_y = shared_pix[threadIdx.y];
            int s_dy = start_distance(pix_y, nx1 % 32);
            merge(labels, line_base, line_base + nx1 - s_dy);
        }
    }
}

// **************************************************************************
template <typename image_t>
__global__
void merge_strip(const image_t *image, int  *labels, int nx, int ny)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny) && (y > 0);
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadActive)
    {
        int k_yx = y * nx + x;
        int k_y1x = k_yx - nx;

        image_t p_yx = image[k_yx];
        image_t p_y1x = image[k_y1x];

        int pix_y = __ballot_sync(activeMask, p_yx);
        int pix_y1 = __ballot_sync(activeMask, p_y1x);

        if (p_yx && p_y1x)
        {
            int s_dy = start_distance(pix_y, threadIdx.x);
            int s_dy1 = start_distance(pix_y1, threadIdx.x);

            if ((s_dy == 0) || (s_dy1 == 0))
                merge(labels, k_yx - s_dy, k_y1x - s_dy1);
        }
    }
}

// **************************************************************************
template <typename image_t, typename label_id_t>
__global__
void relabel(const image_t *image, const int *labels, label_id_t *label_ids, int nx, int ny)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny);
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadActive)
    {
        int k_yx = y * nx + x;
        image_t p_yx = image[k_yx];

        int pix_y = __ballot_sync(activeMask, p_yx);
        int s_dy = start_distance(pix_y, threadIdx.x);

        int label = 0;
        if (p_yx && (s_dy == 0))
        {
            label = labels[k_yx];
            while (label != labels[label])
                label = labels[label];
        }

        label = __shfl_sync(activeMask, label, threadIdx.x - s_dy);

        if (p_yx)
            label_ids[k_yx] = label_ids[label];
    }
}

/** count the number of equivalence trees and assign each an id in 1 to number
 * of trees. this can be used to generate the output label.
 *
 * @param[in] labels the equivalence trees
 * @param[out] label_ids final label ids stored at the roots of their
 *                       equivalence trees
 * @param[in] nx the image size in the x-direction
 * @param[in] ny the image size in the y-direction
 * @param[inout] n_ids the number of trees found
 *
 * n_ids should be set to 1 before calling to include the background at id 0.
 */
template <typename label_id_t>
__global__
void enumerate_equivalences(const int *labels, label_id_t *label_ids,
                            int nx, int ny, int *n_ids)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny);

    if (threadActive)
    {
        int k_yx = y * nx + x;
        int label = labels[k_yx];

        // identify if this index is a root in an equivalence tree get the next
        // available label assign to the tree root
        if (label && (label == k_yx))
            label_ids[label] = atomicAdd(n_ids, 1);
    }
}


/// 2D/3D connected component labeler driver
/**
 * given a binary segmentation(segments) and buffer(components), both
 * with dimensions described by the given exent(ext), compute
 * the labeling.
 *
 * This implementation is based on
 *
 * Arthur Hennequin, Lionel Lacassagne, Laurent Cabaret, Quentin Meunier. A new
 * Direct Connected Component Labeling and Analysis Algorithms for GPUs. 2018
 * Conference on Design and Architectures for Signal and Image Processing
 * (DASIP), Oct 2018, Porto, Portugal.
 *
 * modifications:
 * 1. handles inputs that are not multiples of CUDA warp size.
 * 2. labels correctly across a periodic boundary in the x-direction
 * 3. generates label ids from 1 to num labels
 *
 * limitations:
 * 1. this is a 2D implementation while the CPU version is 3D.
 * 2. label ids are not deterministic, meaning the same blob can have a different
 *    id in different runs.
 * 3. periodic boundary is only implemented in the x-direction. supporting a
 *    periodic bc in the y-direction could be added if needed
 * 4. this is a 32 bit implementation, and can only handle images of total size
 *    2**32-1 Cconversion to 64 bit inorder to support larger images would be
 *    straightforward by changing kernels to operate on long long. Alternatively
 *    for 32 bit implementation a further optimization is processing 2 pixels at a
 *    time using 64 bit instructions
 */
template <typename segment_t, typename component_t>
int label(cudaStream_t strm, unsigned long *ext, int periodic_in_x,
    const segment_t *segments, component_t *components, component_t &n_components)
{
    int nx = ext[1] - ext[0] + 1;
    int ny = ext[3] - ext[2] + 1;
    int nxy = nx*ny;

    // allocate storage for the equivalence table
    hamr::buffer<int> labels(hamr::buffer_allocator::cuda_async, strm,
                             hamr::buffer_transfer::async, nxy, 0);

    // allocate storage for the component count. initialize to 1 since the
    // backgound is included in the count
    hamr::buffer<int> n_comp(hamr::buffer_allocator::cuda_async, strm,
                             hamr::buffer_transfer::async, 1, 1);

    // generate initial lables (horizontal strip decomp)
    int num_strips = ny / STRIP_HEIGHT + (ny % STRIP_HEIGHT ? 1 : 0);

    dim3 blocks(1, num_strips);
    dim3 threads(NUM_THREADS_X, STRIP_HEIGHT);

    label_strip<<<blocks, threads, 0, strm>>>
        (segments, labels.data(), nx, ny, periodic_in_x);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch label kernel. " << cudaGetErrorString(ierr))
        return -1;
    }

    // merge labels (tile decomp)
    int num_tiles = nx / NUM_THREADS_X + (nx % NUM_THREADS_X ? 1 : 0);
    blocks = dim3(num_tiles, num_strips);

    merge_strip<<<blocks, threads, 0, strm>>>(segments, labels.data(), nx, ny);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch merge kernel. " << cudaGetErrorString(ierr))
        return -1;
    }

    // assign each equivalence tree an ordinal id
    enumerate_equivalences<<<blocks, threads, 0, strm>>>
        (labels.data(), components, nx, ny, n_comp.data());

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch enumerate kernel. " << cudaGetErrorString(ierr))
        return -1;
    }

    // resolve equivalences on tiles
    relabel<<<blocks, threads, 0, strm>>>
        (segments, labels.data(), components, nx, ny);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch relabel kernel.")
        return -1;
    }

    // get the count back to the host
    n_comp.get(0, &n_components, 0, 1);
    cudaStreamSynchronize(strm);

    return 0;
}

}
#endif


namespace host_impl {

/// hold i,j,k index triplet
struct id3
{
    id3() : i(0), j(0), k(0) {}
    id3(unsigned long p, unsigned long q, unsigned long r)
        : i(p), j(q), k(r) {}

    unsigned long i;
    unsigned long j;
    unsigned long k;
};

/// 2D/3D connected component labeler
/**
given seed(i0,j0,k0) that's in a component to label, the
current component(current_component), a binary segmentation(segments),
and a set of components(components) of dimensions nx,ny,nz,nxy,
walk the segmentation from the seed labeling it as we go.
when this function returns this component is completely
labeled. this is the 1 pass algorithm.
*/
template <typename segment_t, typename component_t>
void non_periodic_labeler(unsigned long i0, unsigned long j0, unsigned long k0,
    component_t current_component, unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, const segment_t *segments,
    component_t *components)
{
    std::deque<id3> work_queue;
    work_queue.push_back(id3(i0,j0,k0));

    while (work_queue.size())
    {
        id3 ijk = work_queue.back();
        work_queue.pop_back();

        long s0 = ijk.k > 0 ? -1 : 0;
        long s1 = ijk.k < nz-1 ? 1 : 0;
        for (long s = s0; s <= s1; ++s)
        {
            unsigned long ss = ijk.k + s;
            unsigned long kk = ss*nxy;

            long r0 = ijk.j > 0 ? -1 : 0;
            long r1 = ijk.j < ny-1 ? 1 : 0;
            for (long r = r0; r <= r1; ++r)
            {
                unsigned long rr = ijk.j + r;
                unsigned long jj = rr*nx;

                long q0 = ijk.i > 0 ? -1 : 0;
                long q1 = ijk.i < nx-1 ? 1 : 0;
                long q_inc = (r || s) ? 1 : 2;
                for (long q = q0; q <= q1; q += q_inc)
                {
                    unsigned long qq = ijk.i + q;
                    unsigned long w = qq + jj + kk;

                    if (segments[w] && !components[w])
                    {
                        components[w] = current_component;
                        work_queue.push_back(id3(qq,rr,ss));
                    }
                }
            }
        }
    }
}

/// 2D/3D connected component labeler, with periodic boundary in x
/**
given seed(i0,j0,k0) that's in a component to label, the
current component(current_component), a binary segmentation(segments),
and a set of components(components) of dimensions nx,ny,nz,nxy,
walk the segmentation from the seed labeling it as we go.
when this function returns this component is completely
labeled. this is the 1 pass algorithm.

notes:
if we have a periodic bc then neighborhood includes cells -1 to 1, relative
to the current index, else the neighborhood is constrained to 0 to 1, or
-1 to 0.

    long s0 = periodic_in_z ? -1 : ijk.k > 0 ? -1 : 0;
    long s1 = periodic_in_z ? 1 : ijk.k < nz-1 ? 1 : 0;

then when an index goes out of bounds because the neighborhood crosses the
periodic bc

    ss = (ss + nz) % nz;

wraps it around
*/
template <typename segment_t, typename component_t>
void periodic_labeler(unsigned long i0, unsigned long j0, unsigned long k0,
    component_t current_component, unsigned long nx, unsigned long ny,
    unsigned long nz, unsigned long nxy, int periodic_in_x, int periodic_in_y,
    int periodic_in_z, const segment_t *segments,
    component_t *components)
{
    std::deque<id3> work_queue;
    work_queue.push_back(id3(i0,j0,k0));

    while (work_queue.size())
    {
        id3 ijk = work_queue.back();
        work_queue.pop_back();

        long s0 = periodic_in_z ? -1 : ijk.k > 0 ? -1 : 0;
        long s1 = periodic_in_z ? 1 : ijk.k < nz-1 ? 1 : 0;
        for (long s = s0; s <= s1; ++s)
        {
            unsigned long ss = ijk.k + s;
            ss = (ss + nz) % nz;
            unsigned long kk = ss*nxy;

            long r0 = periodic_in_y ? -1 : ijk.j > 0 ? -1 : 0;
            long r1 = periodic_in_y ? 1 : ijk.j < ny-1 ? 1 : 0;
            for (long r = r0; r <= r1; ++r)
            {
                unsigned long rr = ijk.j + r;
                rr = (rr + ny) % ny;
                unsigned long jj = rr*nx;

                long q0 = periodic_in_x ? -1 : ijk.i > 0 ? -1 : 0;
                long q1 = periodic_in_x ? 1 : ijk.i < nx-1 ? 1 : 0;
                long q_inc = (r || s) ? 1 : 2;
                for (long q = q0; q <= q1; q += q_inc)
                {
                    long qq = ijk.i + q;
                    qq = (qq + nx) % nx;
                    unsigned long w = qq + jj + kk;

                    if (segments[w] && !components[w])
                    {
                        components[w] = current_component;
                        work_queue.push_back(id3(qq,rr,ss));
                    }
                }
            }
        }
    }
}

/// 2D/3D connected component labeler driver
/**
 * given a binary segmentation(segments) and buffer(components), both
 * with dimensions described by the given exent(ext), compute
 * the labeling.
 */
template <typename segment_t, typename component_t>
void label(unsigned long *ext, int periodic_in_x, int periodic_in_y,
    int periodic_in_z, const segment_t *segments, component_t *components,
    component_t &max_component)
{
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;
    unsigned long nxy = nx*ny;

    // initialize the components
    component_t current_component = 0;
    memset(components, 0, nxy*nz*sizeof(component_t));

    // visit each element to see if it is a seed
    for (unsigned long k = 0; k < nz; ++k)
    {
        unsigned long kk = k*nxy;
        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jj = j*nx;
            for (unsigned long i = 0; i < nx; ++i)
            {
                unsigned long q = kk + jj + i;

                // found seed, label it
                if (segments[q] && !components[q])
                {
                    components[q] = ++current_component;
                    periodic_labeler(i,j,k, current_component,
                        nx,ny,nz,nxy, periodic_in_x, periodic_in_y,
                        periodic_in_z, segments, components);
                }
            }
        }
    }

    max_component = current_component;
}

};


// --------------------------------------------------------------------------
teca_connected_components::teca_connected_components() :
    component_variable(""), segmentation_variable("")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_connected_components::~teca_connected_components()
{}

// --------------------------------------------------------------------------
std::string teca_connected_components::get_component_variable(
    const teca_metadata &request)
{
    std::string component_var = this->component_variable;
    if (component_var.empty())
    {
        if (request.has("component_variable"))
            request.get("component_variable", component_var);
        else if (this->segmentation_variable.empty())
            component_var = "components";
        else
            component_var = this->segmentation_variable + "_components";
    }
    return component_var;
}


// --------------------------------------------------------------------------
std::string teca_connected_components::get_segmentation_variable(
    const teca_metadata &request)
{
    std::string segmentation_var = this->segmentation_variable;

    if (segmentation_var.empty() &&
        request.has("segmentation_variable"))
            request.get("segmentation_variable",
                segmentation_var);

    return segmentation_var;
}

// --------------------------------------------------------------------------
teca_metadata teca_connected_components::get_output_metadata(
    unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_connected_components::get_output_metadata" << std::endl;
#endif
    (void) port;

    std::string component_var = this->component_variable;
    if (component_var.empty())
    {
        if (this->segmentation_variable.empty())
            component_var = "components";
        else
            component_var = this->segmentation_variable + "_components";
    }

    // tell the downstream about the variable we produce
    teca_metadata md = input_md[0];
    md.append("variables", component_var);

    // add metadata for CF I/O
    teca_metadata atts;
    md.get("attributes", atts);

    teca_metadata seg_var_atts;
    atts.get(this->segmentation_variable, seg_var_atts);

    auto dim_active = teca_array_attributes::xyzt_active();
    seg_var_atts.get("mesh_dim_active", dim_active);

    std::ostringstream oss;
    oss << "the connected components of " << this->segmentation_variable;

    teca_array_attributes cc_atts(
        teca_variant_array_code<short>::get(),
        teca_array_attributes::point_centering, 0, dim_active,
        "unitless", component_var, oss.str().c_str());

    atts.set(component_var, (teca_metadata)cc_atts);

    md.set("attributes", atts);

    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_connected_components::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_connected_components::get_upstream_request" << std::endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    std::string segmentation_var = this->get_segmentation_variable(request);
    if (segmentation_var.empty())
    {
        TECA_FATAL_ERROR("A segmentation variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(segmentation_var);

    // remove fromt the request what we generate
    std::string component_var = this->get_component_variable(request);
    arrays.erase(component_var);

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_connected_components::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_connected_components::execute" << std::endl;
#endif
    (void)port;

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif

    // get the input
    const_p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);
    if (!in_mesh)
    {
        TECA_FATAL_ERROR("empty input, or not a cartesian_mesh")
        return nullptr;
    }

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string segmentation_var = this->get_segmentation_variable(request);
    if (segmentation_var.empty())
    {
        TECA_FATAL_ERROR("A segmentation variable was not specified")
        return nullptr;
    }

    const_p_teca_variant_array input_array
        = out_mesh->get_point_arrays()->get(segmentation_var);
    if (!input_array)
    {
        TECA_FATAL_ERROR("The segmentation variable \"" << segmentation_var
            << "\" is not in the input")
        return nullptr;
    }

    // get mesh dimension
    unsigned long extent[6];
    out_mesh->get_extent(extent);

    unsigned long whole_extent[6];
    out_mesh->get_whole_extent(whole_extent);

    // check for periodic bc.
    int periodic_in_x = 0;
    out_mesh->get_periodic_in_x(periodic_in_x);
    if (periodic_in_x &&
        (extent[0] == whole_extent[0]) && (extent[1] == whole_extent[1]))
        periodic_in_x = 1;

    int periodic_in_y = 0;
    out_mesh->get_periodic_in_y(periodic_in_y);
    if (periodic_in_y &&
        (extent[2] == whole_extent[2]) && (extent[3] == whole_extent[3]))
        periodic_in_y = 1;

    int periodic_in_z = 0;
    out_mesh->get_periodic_in_z(periodic_in_z);
    if (periodic_in_z &&
        (extent[4] == whole_extent[4]) && (extent[5] == whole_extent[5]))
        periodic_in_z = 1;

    // label connected components
    size_t n_elem = input_array->size();

    p_teca_short_array components;
    short *p_components = nullptr;
    short num_components = 0;

#if defined(TECA_HAS_CUDA)
    int device_id = -1;
    request.get("device_id", device_id);

    if (device_id >= 0)
    {
        // can the CUDA implementation handle this input?
        int three_d = (extent[5] - extent[4] + 1) > 1;
        if (three_d || periodic_in_y)
        {
            if (this->get_verbose() && (rank == 0))
            {
                TECA_STATUS("Assigned to CUDA device " << device_id
                    << " but computing on the host because 3D(" << three_d
                    << ") periodic_in_y(" << periodic_in_y << ")")
            }
            device_id = -1;
        }
    }

    if (device_id >= 0)
    {
        std::tie(components, p_components) =
            ::New<teca_short_array>(n_elem, 0, allocator::cuda_async);

        VARIANT_ARRAY_DISPATCH(input_array.get(),

            auto [sp_in, p_in] = get_cuda_accessible<CTT>(input_array);

            if (cuda_impl::label(cudaStreamPerThread, extent, periodic_in_x,
                             p_in, p_components, num_components))
            {
                TECA_FATAL_ERROR("Failed to compute connected component labeling")
                return nullptr;
            }
            )
    }
    else
    {
#endif
        short max_component = 0;
        std::tie(components, p_components) = ::New<teca_short_array>(n_elem);
        VARIANT_ARRAY_DISPATCH(input_array.get(),

            auto [sp_in, p_in] = get_cpu_accessible<CTT>(input_array);

            host_impl::label(extent, periodic_in_x, periodic_in_y,
                periodic_in_z, p_in, p_components, max_component);
            )
        num_components = max_component + 1;
#if defined(TECA_HAS_CUDA)
    }
#endif

    // put components in output
    std::string component_var = this->get_component_variable(request);
    out_mesh->get_point_arrays()->set(component_var, components);

    // put the component ids in the metadata
    p_teca_short_array component_id = teca_short_array::New(num_components);
    short *p_component_id = component_id->data();
    for (short i = 0; i < num_components; ++i)
        p_component_id[i] = i;

    teca_metadata &omd = out_mesh->get_metadata();
    omd.set("component_ids", component_id);
    omd.set("number_of_components", num_components);
    omd.set("background_id", short(0));

    if (this->get_verbose() && (rank == 0))
    {
        TECA_STATUS("connected component labeling computed on "
            << (device_id >= 0 ? "CUDA device" : "the host")
            << "(" << device_id << ")")
    }

    return out_mesh;
}
