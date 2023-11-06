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
#include <chrono>

//#define TECA_DEBUG

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;
using seconds_t = std::chrono::duration<double, std::chrono::seconds::period>;

#if defined(TECA_HAS_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>
#include "hamr_buffer.h"

namespace cuda_impl
{
/// locate the offset from tx to the first connected pixel to the left.
__device__
unsigned int start_distance(unsigned int mask, int tx)
{
    int sd =  __clz(~(mask << (32 - tx)));
    return sd;
}

/// locate the offset from tx to the last connected pixel to the right.
__device__
int end_distance(int mask, int tx)
{
    return __ffs(~(mask >> (tx + 1)));
}

/// swap two values
template <typename T>
__device__
void swap(T &a, T &b)
{
    T c = a;
    a = b;
    b = c;
}

#if !defined(NDEBUG)
/// abort if an invalid label / label address is used
__device__
int assert_valid_label(const int *labels, int l1)
{
    if (!((l1 >= 0) && (labels[l1] >=0 )))
    {
        printf("ERROR: label[%d] = %d\n", l1, labels[l1]);
        return -1;
    }
    return 0;
}
#endif

/** Equate two components. find the root of the two equivalence trees the
 * labels are belonging to and writing the minimum root index to the root with
 * the maximum index.
 */
__device__
void merge(int *labels, int l1, int l2)
{
    while ((l1 != l2) && (l1 != labels[l1]))
    {
#if !defined(NDEBUG)
        if (assert_valid_label(labels, l1)) asm("trap;");
#endif
        l1 = labels[l1];
    }

    while ((l1 != l2) && (l2 != labels[l2]))
    {
#if !defined(NDEBUG)
        if (assert_valid_label(labels, l2)) asm("trap;");
#endif
        l2 = labels[l2];
    }

    while (l1 != l2)
    {
        if (l1 < l2)
            swap(l1, l2);

        int l3 = atomicMin(&labels[l1], l2);

        l1 = ( l1 == l3 ? l2 : l3 );
    }
}

/** 8-connected component labeler, based on the 32 bit implementation of the
 * 4-connected labeler HA4 (https://hal.science/hal-01923784)
 *
 * Detects connected components in horizontal strips of a fixed height
 * STRIP_HEIGHT. The inputs are a bit map where region of interest is defined
 * by mesh cells set to 1 and 0 elsewhere. The output will be an initial
 * labeling of components inside each horizontal strip.  A second pass is
 * needed to resolve equivalences across strips (see merge_strips).  Note that
 * components are labeled with the first flat index of the component. 0 may
 * belong to a label, and does not represent the background in the lables. For
 * this reason labels should be initialized to -1 before calling. For efficiency
 * only the first cell of each horizontal run will be labeled. The label identifies
 * the parent component/node address in the equivalence table. Follow these
 * back to the component/tree root node, labels[addr] == addr.
 *
 * @param[in] image bit map with cells of interest set to 1
 * @param[inout] labels the equivalence table. initialized to -1
 * @param[in] nx the number of cells in the x-direction
 * @param[in] ny the number of cells in the y-direction
 * @param[in] periodic apply a periodic boundary condition in the x-direction.
 *
 */

#define NUM_THREADS_X 32
#define STRIP_HEIGHT 4

template <typename image_t>
__global__
void label_strip(const image_t *image, int *labels, int nx, int ny, bool periodic)
{
    // notation: (c,r) c: 1 is the current pixel, 0 is to its left, 2 is to its right.
    //                 r: 1 is the current pixel, 0 is below it

    __shared__ int s_bts_1[STRIP_HEIGHT];
    __shared__ int s_bts_0[STRIP_HEIGHT];
    __shared__ int s_bts_2[STRIP_HEIGHT];

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    // initialize bit masks
    if (threadIdx.x == 0)
        s_bts_1[threadIdx.y] = 0;

    int line_base = y * nx + x;

    int dist_l1 = 0; // distance from last pixel in the previous iteration to the last connected pixel on its left this line
    int dist_l0 = 0; // distance from last pixel in the previous iteration to the last connected pixel on its left previous line

    int maxI = nx % 32 ? (nx / 32 + 1) * 32 : nx;
    for (int i = 0; i < maxI; i += 32)
    {
        int q11 = line_base + i; // the flat address of the current pixel. notation: q(col)(row)

        // compute masks for warp collectives
        int active_q1 = (((x + i) < nx) && (y < ny)) ? 1 : 0;
        int active_q2 = (((x + i + 32) < nx) && (y < ny)) ? 1 : 0;

        int img_q11 = active_q1 ? image[q11] : 0;           // the image value at q11
        int bts_q11 = __ballot_sync(0xffffffff, img_q11);   // bit mask w. image values around q11 for all 32 thread in this warp

        int img_q21 = active_q2 ? image[q11 + 32] : 0;      // the image value at q21
        int bts_q21 = __ballot_sync(0xffffffff, img_q21);   // bit mask w. image values around q21 for all 32 thread in previous warp

        // initialize row wise equivalence tree roots to their flat address
        int start_q11 = start_distance(bts_q11, threadIdx.x);  // distance from q11 to the last connected pixel on its left
        if (active_q1 && img_q11 && (start_q11 == 0))
        {
            labels[q11] = threadIdx.x == 0 ? q11 - dist_l1 : q11;
        }

        // store the bit masks so that we may later look at masks in other rows in the strip
        if (threadIdx.x == 0)
        {
            s_bts_0[threadIdx.y] = s_bts_1[threadIdx.y]; // get the left side bit masks from the previous iteration
            s_bts_1[threadIdx.y] = bts_q11;
            s_bts_2[threadIdx.y] = bts_q21;
        }

        // wait for other row's masks to become available
        __syncthreads();

        int bts_q10 = threadIdx.y ? s_bts_1[threadIdx.y - 1] : 0; // the mask of the warp directly below
        int bts_q00 = threadIdx.y ? s_bts_0[threadIdx.y - 1] : 0; // mask below and left
        int bts_q20 = threadIdx.y ? s_bts_2[threadIdx.y - 1] : 0; // mask below and right

        // prevent a race between the above conditional read and the next loop iteration.
        // in particular trheadIdx.y == 0 has less work and can continue the loop sooner
        __syncthreads();

        if (img_q11)
        {
            int firstThread = threadIdx.x == 0;
            int secondThread = threadIdx.x == 1;
            int lastThread = threadIdx.x == 31;

            int img_q10 = bts_q10 & (1 << threadIdx.x);             // the pixel directly below
            int start_q10 = start_distance(bts_q10, threadIdx.x);   // offset to q10's equivalence tree node

            if (firstThread)
            {
                start_q11 = dist_l1; // look left into the pixels from the previous iteration, this row
                start_q10 = dist_l0; // look left into the pixels from the previous iteration, row below
            }

            // if pixel directly below is set, sufficient to look there
            // otherwise 2 cases to consider: below left and below right
            if (img_q10)
            {
                if ((start_q11 == 0) || (start_q10 == 0))
                {
                    // equate these two components
                    merge(labels,
                        q11 - start_q11,        // start of the component in this row
                        q11 - nx - start_q10);  // start of the component in the previous row
                }
            }
            else
            {
                // look below and left, threadIdx.y == 0 never does this
                int bts = firstThread ? bts_q00 : bts_q10;
                int img_q00 = bts & (firstThread ? 0x80000000 : (1 << (threadIdx.x - 1)));   // pixel below and left
                if (img_q00)
                {
                    int start_q00 = secondThread ? dist_l0 :
                                    start_distance(bts, firstThread ? 31 : threadIdx.x - 1); // distance to q00 equivalence tree node

                    // equate these two components
                    if ((start_q11 == 0) || (start_q00 == 0))
                    {
                        merge(labels, q11 - start_q11, q11 - nx - 1 - start_q00);
                    }
                }

                // look below and right. threadIdx.y == 0 never does this
                bts  = lastThread ? bts_q20 : bts_q10;
                int img_q20 = bts & (lastThread ? 0x1 : (1 << (threadIdx.x + 1)));           // pixel below and right
                if (img_q20)
                {
                    // by construction (!img_q10 && img_q20) we know q20 is a
                    // node in an equivalence tree.
                    int q20 =  q11 - nx + 1;

                    // initialize the lable at q20 since we work left to right
                    // sequentially it might not have been initialized at this
                    // point.
                    if (lastThread)
                        atomicExch(&labels[q20], q20);

                    // equate these two components
                    merge(labels, q11 - start_q11, q20);
                }
            }
        }

        // store the start of the run for this row and previous row. used by
        // thread 0 if set to locate parent node in the equivalence table.
        int d = start_distance(bts_q10, 32);
        dist_l0 = d == 32 ? d + dist_l0 : d;

        d = start_distance(bts_q11, 32);
        dist_l1 = d == 32 ? d + dist_l1 : d;
    }

    // apply the periodic boundary condition in the x-direction
    int q11 = line_base;            // address of the current pixel
    if (periodic && (threadIdx.x == 0) && (x < nx) && (y < ny) && image[q11])
    {
        int nx1 = nx - 1;           // offset to last element in this row
        int bit_nx1 = nx1 % 32;     // bit containing info about this element

        int bts_qn1 = s_bts_1[threadIdx.y];
        int img_qn1 = bts_qn1 & (1 << bit_nx1); // the value of the pixel at the far end of this row

        int bts_qn0 = threadIdx.y ? s_bts_1[threadIdx.y - 1] : 0;
        int img_qn0 = bts_qn0 & (1 << bit_nx1); // the value of the pixel at the far end of the previous row

        int bts_qn2 = threadIdx.y < STRIP_HEIGHT - 2 ? s_bts_1[threadIdx.y + 1] : 0;
        int img_qn2 = bts_qn2 & (1 << bit_nx1); // the value of the pixel at the far end of the next row

        if (img_qn1)
        {
            // equate this pixel and the one at the far end of the row
            int start_qn1 = start_distance(bts_qn1, bit_nx1);
            merge(labels, q11, q11 + nx1 - start_qn1);
        }
        else if (img_qn0)
        {
            // equate this pixel and the one at the far end of the previous row
            int start_qn0 = start_distance(bts_qn0, bit_nx1);
            merge(labels, q11, q11 - 1 - start_qn0);
        }
        else if (img_qn2)
        {
            // equate this pixel and the one at the far end of the next row
            int start_qn2 = start_distance(bts_qn2, bit_nx1);
            merge(labels, q11, q11 + nx + nx1 - start_qn2);
        }
    }
}

template <typename image_t>
__global__
void merge_strip(const image_t *image, int *labels, int nx, int ny)
{
    // notation: (c,r) c: 1 is the current pixel, 0 is to its left, 2 is to its right.
    //                 r: 1 is the current pixel, 0 is below it

    int y = STRIP_HEIGHT * (blockDim.y * blockIdx.y + threadIdx.y);
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool active_q11 = (x < nx) && (y < ny) && (y > 0);
    bool active_q00 = ((x - 32) >= 0) && ((x - 32) < nx) && (y < ny) && (y > 0);
    bool active_q20 = ((x + 32) < nx) && (y < ny) && (y > 0);

    int q11 = y * nx + x; // the flat address of the current pixel
    int q10 = q11 - nx;   // the falt address of the pixel directly below

    image_t img_q11 = active_q11 ? image[q11] : 0;
    image_t img_q10 = active_q11 ? image[q10] : 0;

    int bts_q11 = __ballot_sync(0xffffffff, img_q11);
    int bts_q10 = __ballot_sync(0xffffffff, img_q10);

    int start_q11 = start_distance(bts_q11, threadIdx.x);

    image_t img_q00 = active_q00 ? image[q10 - 32] : 0;
    image_t img_q20 = active_q20 ? image[q10 + 32] : 0;

    int bts_q00 = __ballot_sync(0xffffffff, img_q00);
    int bts_q20 = __ballot_sync(0xffffffff, img_q20);

    if (img_q11)
    {
        // directly below
        if (img_q10)
        {
            int start_q10 = start_distance(bts_q10, threadIdx.x);

            if ((start_q11 == 0) || (start_q10 == 0))
                merge(labels, q11 - start_q11, q10 - start_q10);
        }
        else
        {
            int firstThread = threadIdx.x == 0;
            int lastThread = (threadIdx.x == 31);

            // below and left
            int img_q00 = firstThread ? (bts_q00 & 0x80000000) : (bts_q10 & (1 << (threadIdx.x - 1)));
            if (img_q00)
            {
                int start_q00 = start_distance(firstThread ? bts_q00 : bts_q10,
                                               firstThread ? 31 : threadIdx.x - 1);

                if ((start_q11 == 0) || (start_q00 == 0))
                    merge(labels, q11 - start_q11, q10 - 1 - start_q00);
            }

            // below and right.  note that if (!img_q10 && img_q20) q20 is a
            // node in the equivalence tree. equate q11 and q20
            int img_q20 = lastThread ? (bts_q20 & 0x1) : (bts_q10 & (1 << (threadIdx.x + 1)));
            if (img_q20)
                merge(labels, q11 - start_q11, q10 + 1);
        }
    }
}


/** visit each pixel and if set fill in with its final label.
 * @param[in] image bit map with regions of interest set
 * @param[in] labels equivalence table identifying the connected components
 * @param[inout] label_ids table with equivalence tree roots set to their final label
 */
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
        int q11 = y * nx + x;
        image_t img_q11 = image[q11]; // value of the current pixel

        int bts_q11 = __ballot_sync(activeMask, img_q11);
        int start_q11 = start_distance(bts_q11, threadIdx.x); // offset to first set pixel to the left in this row

        // get the root of the equivalence tree for this pixel
        int label = 0;
        if (img_q11 && (start_q11 == 0))
        {
            label = labels[q11];
            while (label != labels[label])
            {
#if !defined(NDEBUG)
                assert_valid_label(labels, label);
#endif
                label = labels[label];
            }
        }

        label = __shfl_sync(activeMask, label, threadIdx.x - start_q11);

        // update label to the root
        if (img_q11)
            label_ids[q11] = label_ids[label];
    }
}

/** count the number of equivalence trees and assign each an id in 1 to number
 * of trees. this can be used to generate the final labels from the equivalence
 * table.
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
        int q11 = y * nx + x;
        int label = labels[q11];

        // if this index is a root in an equivalence tree get the next
        // available label assign to the tree root. checking all entries in the
        // table, so we'll need to handle invalid entries
        if ((label >= 0) && (label == q11))
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
 * Modifications:
 * 1. handles inputs that are not multiples of CUDA warp size.
 * 2. labels correctly across a periodic boundary in the x-direction
 * 3. generates label ids from 1 to num labels
 * 4. 8-connected labeling is implemented rather than 4-connected
 *
 * Notes:
 * 1. this is a 2D implementation
 * 2. this is a 32 bit implementation, and can only handle images of total size
 *    2**32-1 Cconversion to 64 bit in order to support larger images would be
 *    straightforward by changing equivalence tables to long long. Alternatively
 *    a further optimization is possible by processing 2 pixels at a time using
 *    64 bit instructions
 * 3. label ids are not deterministic, meaning the same blob can have a different
 *    id in different runs.
 *
 * @param[in] strm          the CUDA stream to submit work on
 * @param[in] ext           the image extents [i0, i1, j0, j1]
 * @param[in] periodic_in_x a flag the denotes a periodic boundary at the
 *                          left/right edge of the image
 * @param[in] segments      the segmented image to label
 * @param[out] components   the labeled image
 * @param[out] n_components the number of connected components
 * @returns 0 if successful
 */
template <typename segment_t, typename component_t>
int label(cudaStream_t strm, unsigned long *ext, int periodic_in_x,
    const segment_t *segments, component_t *components, component_t &n_components)
{
    int nx = ext[1] - ext[0] + 1;
    int ny = ext[3] - ext[2] + 1;
    int nxy = nx*ny;

    // allocate storage for the equivalence table. intialize to -1.
    hamr::buffer<int> labels(hamr::buffer_allocator::cuda_async, strm,
                             hamr::buffer_transfer::async, nxy, -1);

    // allocate storage for the component count. initialize to 1 since the
    // backgound is included in the count
    hamr::buffer<int> n_comp(hamr::buffer_allocator::cuda_async, strm,
                             hamr::buffer_transfer::async, 1, 1);

    // generate initial lables. partition the mesh into horizontal strips.
    // each line is processed by a single thread, in warp size blocks.
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

    // merge labels. we only touch the strip boundary, and address each strip
    // by its block id. this is why number of threads in the y-direction is 1
    int num_tiles = nx / NUM_THREADS_X + (nx % NUM_THREADS_X ? 1 : 0);

    blocks = dim3(num_tiles, num_strips);
    threads = dim3(NUM_THREADS_X, 1);

    merge_strip<<<blocks, threads, 0, strm>>>(segments, labels.data(), nx, ny);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch merge kernel. " << cudaGetErrorString(ierr))
        return -1;
    }

    // assign each equivalence tree an ordinal id which becomes the final
    // label. from here on out use a full 2D decomp
    threads = dim3(NUM_THREADS_X, STRIP_HEIGHT);

    enumerate_equivalences<<<blocks, threads, 0, strm>>>
        (labels.data(), components, nx, ny, n_comp.data());

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to launch enumerate kernel. " << cudaGetErrorString(ierr))
        return -1;
    }

    // assign the final labels.
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
void teca_connected_components::get_component_variable(std::string &var) const
{
    var = this->component_variable;
    if (var.empty())
    {
        var = this->get_segmentation_variable() + "_components";
    }
}

// --------------------------------------------------------------------------
void teca_connected_components::get_segmentation_variable(std::string &var) const
{
    var = this->segmentation_variable;
    if (var.empty())
    {
        TECA_FATAL_ERROR("The segmentation_variable was not specified")
    }
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

    std::string segmentation_var;
    std::string component_var;

    this->get_segmentation_variable(segmentation_var);
    this->get_component_variable(component_var);

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
    oss << "the connected components of " << segmentation_var;

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
    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(segmentation_var);

    // remove fromt the request what we generate
    std::string component_var;
    this->get_component_variable(component_var);
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
    (void)request;

    std::chrono::high_resolution_clock::time_point t0, t1, tcc0, tcc1;
    t0 = std::chrono::high_resolution_clock::now();

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

    unsigned long step = 0;
    in_mesh->get_time_step(step);

    // create output and copy metadata, coordinates, etc
    p_teca_cartesian_mesh out_mesh = teca_cartesian_mesh::New();

    out_mesh->shallow_copy(
        std::const_pointer_cast<teca_cartesian_mesh>(in_mesh));

    // get the input array
    std::string segmentation_var;
    this->get_segmentation_variable(segmentation_var);

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

    int device_id = -1;
#if defined(TECA_HAS_CUDA)
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
        if (teca_cuda_util::set_device(device_id))
            return nullptr;

        std::tie(components, p_components) =
            ::New<teca_short_array>(n_elem, 0, allocator::cuda_async);

        VARIANT_ARRAY_DISPATCH(input_array.get(),

            auto [sp_in, p_in] = get_cuda_accessible<CTT>(input_array);

            tcc0 = std::chrono::high_resolution_clock::now();

            if (cuda_impl::label(cudaStreamPerThread, extent, periodic_in_x,
                                 p_in, p_components, num_components))
            {
                TECA_FATAL_ERROR("Failed to compute connected component labeling")
                return nullptr;
            }

            tcc1 = std::chrono::high_resolution_clock::now();
            )
    }
    else
    {
#endif
        short max_component = 0;
        std::tie(components, p_components) = ::New<teca_short_array>(n_elem);
        VARIANT_ARRAY_DISPATCH(input_array.get(),

            auto [sp_in, p_in] = get_host_accessible<CTT>(input_array);

            sync_host_access_any(input_array);

            tcc0 = std::chrono::high_resolution_clock::now();

            host_impl::label(extent, periodic_in_x, periodic_in_y,
                periodic_in_z, p_in, p_components, max_component);

            tcc1 = std::chrono::high_resolution_clock::now();
            )
        num_components = max_component + 1;
#if defined(TECA_HAS_CUDA)
    }
#endif

    // put components in output
    std::string component_var;
    this->get_component_variable(component_var);
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

    t1 = std::chrono::high_resolution_clock::now();
    seconds_t dt(t1 - t0);
    seconds_t dtcc(tcc1 - tcc0);

    if (this->get_verbose() && (rank == 0))
    {
        TECA_STATUS("connected component labeling for step " << step
            << " computed on " << (device_id >= 0 ? "CUDA device" : "the host")
            << "(" << device_id << ") " << num_components
            << " labels in " << dt.count() << "(" << dtcc.count()
            << " cc) seconds")
    }

    return out_mesh;
}
