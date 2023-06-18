#include "teca_config.h"
#include "teca_common.h"

#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>


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


// requires block height shared memory elements

#define ALL_THREADS 0xffffffff
#define NUM_THREADS_X 32
#define STRIP_HEIGHT 4

__global__
void label_strip(int *image, int *labels, int nx, int ny, bool periodic)
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
        int p_yx = threadActive ? image[k_yx] : 0;
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

__global__
void merge_strip(int *image, int  *labels, int nx, int ny)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny) && (y > 0);
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadActive)
    {
        int k_yx = y * nx + x;
        int k_y1x = k_yx - nx;

        int p_yx = image[k_yx];
        int p_y1x = image[k_y1x];

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


__global__
void relabel(int *image, int *labels, int nx, int ny)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    bool threadActive = (x < nx) && (y < ny);
    unsigned int activeMask = __ballot_sync(0xffffffff, threadActive);

    if (threadActive)
    {
        int k_yx = y * nx + x;
        int p_yx = image[k_yx];

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
            labels[k_yx] = label;
    }
}

void print(const hamr::buffer<int> img, int nx, int ny)
{
    auto [spimg, pimg] = hamr::get_cpu_accessible(img);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            if (pimg[j*nx + i])
                std::cerr << std::setw(4) << pimg[j*nx + i];
            else
                std::cerr << std::setw(4) << " ";;
        }
        std::cerr << std::endl;
    }
}

void print(const hamr::buffer<int> &img, const hamr::buffer<int> &lab, int nx, int ny)
{
    auto [spimg, pimg] = hamr::get_cpu_accessible(img);
    auto [splab, plab] = hamr::get_cpu_accessible(lab);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            if (pimg[j*nx + i])
            {
                if (plab[j*nx + i])
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



int main(int argc, char **argv)
{
    bool periodicBc = true;

    auto cpu_alloc = hamr::buffer_allocator::malloc;
    auto dev_alloc = hamr::buffer_allocator::cuda;

    int nx = 37;
    int ny = 18;

    int pixels[] = {
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
        1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    };

    hamr::buffer<int> image(dev_alloc, nx*ny, pixels);
    hamr::buffer<int> labels(dev_alloc, nx*ny, 0);

    std::cerr << "input:" << std::endl;
    print(image, nx, ny);
    std::cerr << std::endl;

    /*if (nx % NUM_THREADS_X)
    {
        TECA_FATAL_ERROR("The mesh width " << nx
            << " must be a multiple of " << NUM_THREADS_X)
    }

    if (ny % STRIP_HEIGHT)
    {
        TECA_FATAL_ERROR("The mesh height " << ny
            << " must be a multiple of " << STRIP_HEIGHT)
    }*/

    int num_strips = ny / STRIP_HEIGHT + (ny % STRIP_HEIGHT ? 1 : 0);

    dim3 blocks(1, num_strips);
    dim3 threads(NUM_THREADS_X, STRIP_HEIGHT);

    label_strip<<<blocks, threads>>>(image.data(), labels.data(), nx, ny, periodicBc);
    //cudaDeviceSynchronize();

    std::cerr << "labels:" << std::endl;
    print(image, labels, nx, ny);
    std::cerr << std::endl;

    int num_tiles = nx / NUM_THREADS_X + (nx % NUM_THREADS_X ? 1 : 0);
    blocks = dim3(num_tiles, num_strips);

    merge_strip<<<blocks, threads>>>(image.data(), labels.data(), nx, ny);
    //cudaDeviceSynchronize();

    std::cerr << "merged:" << std::endl;
    std::cerr.width(3);
    print(image, labels, nx, ny);
    std::cerr << std::endl;

    relabel<<<blocks, threads>>>(image.data(), labels.data(), nx, ny);
    //cudaDeviceSynchronize();

    std::cerr << "relabel:" << std::endl;
    print(image, labels, nx, ny);
    std::cerr << std::endl;

    return 0;
}
