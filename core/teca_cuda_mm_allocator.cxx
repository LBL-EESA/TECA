#include "teca_cuda_mm_allocator.h"
#include "teca_common.h"

#if !defined(TECA_HAS_CUDA)
// --------------------------------------------------------------------------
p_teca_cuda_mm_allocator teca_cuda_mm_allocator::New()
{
    TECA_ERROR("Failed to allocate memory because CUDA is not available")
    return nullptr;
}

// --------------------------------------------------------------------------
p_teca_allocator teca_cuda_mm_allocator::new_instance() const
{
    TECA_ERROR("Failed to allocate memory because CUDA is not available")
    return nullptr;
}

// --------------------------------------------------------------------------
void *teca_cuda_mm_allocator::do_allocate(std::size_t n_bytes, std::size_t align)
{
    (void) n_bytes;
    (void) align;

    TECA_ERROR("Failed to allocate memory because CUDA is not available")
    return nullptr;
}

// --------------------------------------------------------------------------
void teca_cuda_mm_allocator::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) ptr;
    (void) n_bytes;
    (void) align;

    TECA_ERROR("Failed to de-allocate memory because CUDA is not available")
}

// --------------------------------------------------------------------------
bool teca_cuda_mm_allocator::do_is_equal(const memory_resource_t& other) const noexcept
{
    (void) other;
    TECA_ERROR("Failed to compare resource because CUDA is not available")
    return false;
}
#endif
