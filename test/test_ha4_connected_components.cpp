#include "teca_config.h"
#include "teca_common.h"

#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>

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

        // prevent a race between the above conditional read and theee next loop iteration.
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
    }
}

template <typename image_t>
__global__
void merge_strip(const image_t *image, int  *labels, int nx, int ny)
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


/** count the number of equivalence trees and save their roots.
 *
 * @param[in] labels the equivalence trees
 * @param[in] n_elem the image size
 * @param[inout] ulabels pre-allocated memory at least
 *               n_elem / 2 + n_elem % 2 : 1 0 elements long where
 *               tree roots are stored
 * @param[inout] n_ulabels the number of trees found
 *
 * note: ulabels[0] can be set 0 and n_ulabels set to 1 before calling to
 * include the background label.
 */
__global__
void enumerate_equivalences(const int *labels, int *label_ids, int nx, int ny,
                            int *ulabels, int *n_ulabels)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny);

    if (threadActive)
    {
        int q11 = y * nx + x;
        int label = labels[q11];

        // find root in equivalence tree
        if ((label >= 0) && (label == q11))
        {
            // add to the list of unique labels
            int idx = atomicAdd(n_ulabels, 1);
            ulabels[idx] = label;

            // save the label id
            label_ids[label] = idx;
        }
    }
}



template <typename image_t>
void print(const hamr::buffer<image_t> img, int nx, int ny)
{
    auto [spimg, pimg] = hamr::get_host_accessible(img);
    img.synchronize();

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            if (pimg[j*nx + i])
                std::cerr << std::setw(4) << (int)pimg[j*nx + i];
            else
                std::cerr << std::setw(4) << " ";
        }
        std::cerr << std::endl;
    }
}

template <typename image_t>
void print(const hamr::buffer<image_t> &img, const hamr::buffer<int> &lab, int nx, int ny)
{
    auto [spimg, pimg] = hamr::get_host_accessible(img);
    auto [splab, plab] = hamr::get_host_accessible(lab);

    img.synchronize();

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            if (pimg[j*nx + i])
            {
                if (plab[j*nx + i] >= 0)
                    std::cerr << std::setw(4) << plab[j*nx + i];
                else
                    std::cerr << std::setw(4) << "***";
            }
            else
            {
                std::cerr << std::setw(4) << " ";
            }
        }
        std::cerr << std::endl;
    }
}


hamr::buffer<char> generate_image(hamr::buffer_allocator alloc)
{
    int nx = 37;
    int ny = 18;

    char pixels[] = {
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
        0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
        1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
        1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    };

    long nxy = nx*ny;
    hamr::buffer<char> image(alloc, nxy, pixels);

    return image;
}

hamr::buffer<char> generate_image(int nx, int ny, int max_len, int seed, int n_blobs)
{
    long nxy = nx*ny;
    hamr::buffer<char> image(hamr::buffer_allocator::cuda_host, nxy, '\0');
    char *p_image = image.data();

    srand(seed);

    for (int q = 0; q < n_blobs; ++q)
    {
        int i0 = rand() % nx;
        int j0 = rand() % ny;

        int i_len = rand() % max_len;
        int j_len = rand() % max_len;

        int i1 = std::min(nx - 1, i0 + i_len);
        int j1 = std::min(ny - 1, j0 + j_len);

        int c = 1;

        for (int j = j0; j <= j1; ++j)
        {
            for (int i = i0; i <= i1; ++i)
            {
                p_image[ j * nx + i ] = c;
            }
        }
    }

    image.move(hamr::buffer_allocator::cuda);

    return image;
}

template <typename T>
void write_image(const char *fn, const hamr::buffer<T> &image)
{
    FILE *fh = fopen(fn, "wb");

    auto [spi, pi] = hamr::get_host_accessible(image);
    image.synchronize();

    fwrite(pi, image.size(), sizeof(T), fh);
    fclose(fh);
}




int main(int argc, char **argv)
{
    auto host_alloc = hamr::buffer_allocator::cuda_host;
    auto dev_alloc = hamr::buffer_allocator::cuda;

#if defined(TEST)
    (void)argc;
    (void)argv;

    bool periodicBc = true;

    int nx = 37;
    int ny = 18;

    long nxy = nx*ny;

    hamr::buffer<char> image = generate_image(dev_alloc);
#else
    if (argc != 7)
    {
        std::cerr << "usage: a.out [nx] [ny] [max len] [seed] [num blobs] [periodic]" << std::endl;
        return -1;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int max_len = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int n_blobs = atoi(argv[5]);
    bool periodicBc = atoi(argv[6]);

    long nxy = nx*ny;

    hamr::buffer<char> image = generate_image(nx, ny, max_len, seed, n_blobs);
#endif

    hamr::buffer<int> labels(dev_alloc, nxy, -1);
    hamr::buffer<int> label_ids(dev_alloc, nxy, 0);
    hamr::buffer<int> ulabels(dev_alloc, nxy / 2 + (nxy % 2 ? 1 : 0), 0);
    hamr::buffer<int> nulabels(dev_alloc, 1, 1);

    if (nx < 64)
    {
        std::cerr << "input:" << std::endl;
        print(image, nx, ny);
        std::cerr << std::endl;
    }
    else
    {
        write_image("image.raw", image);
    }

    int num_strips = ny / STRIP_HEIGHT + (ny % STRIP_HEIGHT ? 1 : 0);

    dim3 blocks(1, num_strips);
    dim3 threads(NUM_THREADS_X, STRIP_HEIGHT);

    label_strip<<<blocks, threads>>>(image.data(), labels.data(), nx, ny, periodicBc);

    if (nx < 64)
    {
        std::cerr << "labels:" << std::endl;
        print(image, labels, nx, ny);
        std::cerr << std::endl;
    }
    else
    {
        write_image("labels.raw", labels);
    }

    int num_tiles = nx / NUM_THREADS_X + (nx % NUM_THREADS_X ? 1 : 0);
    blocks = dim3(num_tiles, num_strips);
    threads = dim3(NUM_THREADS_X, 1);

    merge_strip<<<blocks, threads>>>(image.data(), labels.data(), nx, ny);

    if (nx < 64)
    {
        std::cerr << "merged:" << std::endl;
        std::cerr.width(3);
        print(image, labels, nx, ny);
        std::cerr << std::endl;
    }
    else
    {
        write_image("merged.raw", labels);
    }

    threads = dim3(NUM_THREADS_X, STRIP_HEIGHT);

    enumerate_equivalences<<<blocks, threads>>>(labels.data(), label_ids.data(), nx, ny, ulabels.data(), nulabels.data());

    relabel<<<blocks, threads>>>(image.data(), labels.data(), label_ids.data(), nx, ny);

    if (nx < 64)
    {
       std::cerr << "relabel:" << std::endl;
       print(image, label_ids, nx, ny);
       std::cerr << std::endl;
    }
    else
    {
        write_image("relabel.raw", label_ids);
    }

    cudaStreamSynchronize(cudaStreamPerThread);

    int nu = 0;
    nulabels.get(0, &nu, 0, 1);

    ulabels.resize(nu);
    ulabels.move(host_alloc);
    ulabels.synchronize();

    ulabels.print();

    return 0;
}
