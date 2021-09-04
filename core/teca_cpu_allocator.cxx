#include "teca_cpu_allocator.h"
#include "teca_common.h"

#include <memory_resource>
#include <iostream>

// --------------------------------------------------------------------------
p_teca_cpu_allocator teca_cpu_allocator::New()
{
    return std::shared_ptr<teca_cpu_allocator>(new teca_cpu_allocator);
}

// --------------------------------------------------------------------------
p_teca_allocator teca_cpu_allocator::new_instance() const
{
    return std::shared_ptr<teca_cpu_allocator>(new teca_cpu_allocator);
}

// --------------------------------------------------------------------------
void *teca_cpu_allocator::do_allocate(std::size_t n_bytes, std::size_t align)
{
    void *ptr = aligned_alloc(align, n_bytes);
    if (ptr == 0)
    {
        TECA_ERROR("Failed to allocate " << n_bytes << " aligned to "
            << align << " bytes")

        throw std::bad_alloc();
    }

    if (this->verbose > 1)
    {
        std::cerr << "teca_cpu_allocator(" << this << ") allocated " << n_bytes
            << " alligned to " << align << " bytes at " << ptr << std::endl;
    }

    return ptr;
}

// --------------------------------------------------------------------------
void teca_cpu_allocator::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) n_bytes;
    (void) align;

    free(ptr);

    if (this->verbose > 1)
    {
        std::cerr << "teca_cpu_allocator(" << this << ") deallocated "
            << n_bytes << std::endl;
    }
}

// --------------------------------------------------------------------------
bool teca_cpu_allocator::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return dynamic_cast<const teca_cpu_allocator*>(&other) != nullptr;
}
