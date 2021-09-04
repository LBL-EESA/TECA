#ifndef teca_cpu_allocator_h
#define teca_cpu_allocator_h

#include "teca_allocator.h"
#include <memory>

class teca_cpu_allocator;
using p_teca_cpu_allocator = std::shared_ptr<teca_cpu_allocator>;

/// an allocator that manages memory for use only on the CPU
class teca_cpu_allocator : public teca_allocator
{
public:
    static p_teca_cpu_allocator New();

    /// return a new instance of the same type of allocator
    p_teca_allocator new_instance() const override;

    /// returns true if the memory can be use on the CPU
    bool cpu_accessible() const override { return true; }

    /// returns true if the memory can be use on the GPU from CUDA
    bool cuda_accessible() const override { return false; }

    /// return the name of the class
    const char *get_class_name() const override
    { return "teca_cpu_allocator"; }

protected:
    teca_cpu_allocator() {}

private:
    /// allocate memory for use on the CPU
    void *do_allocate(std::size_t n_bytes, std::size_t alignment) override;

    /// deallocate memory allocated for use on the CPU
    void do_deallocate(void *ptr, std::size_t n_bytes, std::size_t alignment) override;

    /// check for equality (eqaul if one can delete the memory allocated by the other)
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept;
};

#endif
