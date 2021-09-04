#include "teca_cuda_mm_allocator.h"
#include "teca_common.h"

#include <memory_resource>
#include <iostream>

// --------------------------------------------------------------------------
p_teca_cuda_mm_allocator teca_cuda_mm_allocator::New()
{
    return std::shared_ptr<teca_cuda_mm_allocator>(new teca_cuda_mm_allocator);
}

// --------------------------------------------------------------------------
p_teca_allocator teca_cuda_mm_allocator::new_instance() const
{
    return std::shared_ptr<teca_cuda_mm_allocator>(new teca_cuda_mm_allocator);
}

// --------------------------------------------------------------------------
void *teca_cuda_mm_allocator::do_allocate(std::size_t n_bytes, std::size_t align)
{
    (void) align;

    void *ptr = nullptr;

    cudaError_t ierr = cudaMallocManaged(&ptr, n_bytes, cudaMemAttachGlobal);
    if (ierr != cudaSuccess)
    {
        TECA_ERROR("Failed to allocate " << n_bytes << " of CUDA managed memory. "
            << cudaGetErrorString(ierr))

        throw std::bad_alloc();
    }

    if (this->verbose > 1)
    {
        std::cerr << "teca_cuda_mm_allocator(" << this << ") allocated " << n_bytes
            << " alligned to " << align << " byte boundary at " << ptr << std::endl;
    }

    return ptr;
}

// --------------------------------------------------------------------------
void teca_cuda_mm_allocator::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) n_bytes;
    (void) align;

    cudaError_t ierr = cudaFree(ptr);

    if (ierr != cudaSuccess)
    {
        TECA_ERROR("Failed to free " << n_bytes << " of CUDA managed memory at "
            << ptr << ". " << cudaGetErrorString(ierr))
    }

    if (this->verbose > 1)
    {
        std::cerr << "teca_cuda_mm_allocator(" << this << ") deallocated " << n_bytes
            << " alligned to " << align << " byte boundary  at " << ptr << std::endl;
    }
}

// --------------------------------------------------------------------------
bool teca_cuda_mm_allocator::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return dynamic_cast<const teca_cuda_mm_allocator*>(&other) != nullptr;
}

