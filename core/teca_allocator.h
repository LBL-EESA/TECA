#ifndef teca_allocator_h
#define teca_allocator_h

#if defined(__clang__)
#include <experimental/memory_resource>
using memory_resource_t = std::experimental::pmr::memory_resource;
#else
#include <memory_resource>
using memory_resource_t = std::pmr::memory_resource;
#endif

#include <memory>

class teca_allocator;
using p_teca_allocator = std::shared_ptr<teca_allocator>;

/// base class for allocators
class teca_allocator : public memory_resource_t, std::enable_shared_from_this<teca_allocator>
{
public:
    teca_allocator() : verbose(0) {}
    virtual ~teca_allocator() {}

    /// return a new instance of the same type of allocator
    virtual p_teca_allocator new_instance() const = 0;

    /// returns true if the memory can be used by code running on the CPU
    virtual bool cpu_accessible() const { return false; }

    /// returns true if the memory can be use on the GPU from CUDA
    virtual bool cuda_accessible() const { return false; }

    /// return the name of the class
    virtual const char *get_class_name() const = 0;

    /// return the current verbosity level
    int get_verbose() { return this->verbose; }

    /// set the verbosity level
    void set_verbose(int val) { this->verbose = val; }

protected:
    int verbose;
};

#endif
