#include "array.h"
#include "teca_binary_stream.h"
#include "teca_bad_cast.h"

#include <hamr_buffer.h>
#include <hamr_buffer_pointer.h>

#include <utility>
#include <iostream>
#include <sstream>

using std::ostringstream;
using std::ostream;

struct array::array_internals
{
    array_internals() {}
    hamr::p_buffer<double> buffer;
};

// --------------------------------------------------------------------------
array::array() : extent({0,0})
{
    this->internals = new array_internals;
}

// --------------------------------------------------------------------------
array::~array()
{
    delete this->internals;
}

// --------------------------------------------------------------------------
array::array(allocator alloc) : extent({0,0})
{
    this->internals = new array_internals;

    this->internals->buffer =
        std::make_shared<hamr::buffer<double>>(alloc);
}

// --------------------------------------------------------------------------
size_t array::size() const
{
    if (this->internals->buffer)
        return this->internals->buffer->size();

    return 0;
}

// --------------------------------------------------------------------------
p_array array::new_host_accessible()
{
    return array::New(allocator::malloc);
}

// --------------------------------------------------------------------------
p_array array::new_cuda_accessible()
{
    return array::New(allocator::cuda_async);
}

// --------------------------------------------------------------------------
p_array array::New(allocator alloc)
{
    return p_array(new array(alloc));
}

// --------------------------------------------------------------------------
bool array::host_accessible() const
{
    return this->internals->buffer->host_accessible();
}

// --------------------------------------------------------------------------
bool array::cuda_accessible() const
{
    return this->internals->buffer->cuda_accessible();
}

// --------------------------------------------------------------------------
p_teca_dataset array::new_copy(allocator alloc) const
{
    p_teca_dataset a(new array(alloc));
    a->copy(this->shared_from_this());
    return a;
}

// --------------------------------------------------------------------------
p_teca_dataset array::new_shallow_copy()
{
    p_teca_dataset a(new array());
    a->shallow_copy(this->shared_from_this());
    return a;
}

// --------------------------------------------------------------------------
std::shared_ptr<const double> array::get_host_accessible() const
{
    return this->internals->buffer->get_host_accessible();
}

// --------------------------------------------------------------------------
std::shared_ptr<const double> array::get_cuda_accessible() const
{
    return this->internals->buffer->get_cuda_accessible();
}

// --------------------------------------------------------------------------
double *array::data()
{
    return this->internals->buffer->data();
}


// --------------------------------------------------------------------------
void array::resize(size_t n)
{
    this->internals->buffer->resize(n, 0.0);
    this->extent = {0, n};
}

// --------------------------------------------------------------------------
void array::clear()
{
    this->internals->buffer->free();
    this->extent[0] = 0;
    this->extent[1] = 0;
}

// --------------------------------------------------------------------------
void array::copy(const const_p_teca_dataset &other, allocator alloc)
{
    const_p_array other_a = std::dynamic_pointer_cast<const array>(other);

    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    if (this == other_a.get())
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;

    this->internals->buffer =
        std::make_shared<hamr::buffer<double>>
            (alloc, ref_to(other_a->internals->buffer));
}

// --------------------------------------------------------------------------
void array::shallow_copy(const p_teca_dataset &other)
{
    const_p_array other_a = std::dynamic_pointer_cast<const array>(other);

    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    if (this == other_a.get())
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;
    this->internals->buffer = other_a->internals->buffer;
}

// --------------------------------------------------------------------------
void array::copy_metadata(const const_p_teca_dataset &other)
{
    const_p_array other_a = std::dynamic_pointer_cast<const array>(other);
    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    if (this == other_a.get())
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;
    this->internals->buffer->resize(this->extent[1]-this->extent[0]);
}

// --------------------------------------------------------------------------
void array::swap(const p_teca_dataset &other)
{
    p_array other_a = std::dynamic_pointer_cast<array>(other);
    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    std::swap(this->name, other_a->name);
    std::swap(this->extent, other_a->extent);
    std::swap(this->internals, other_a->internals);
}

// --------------------------------------------------------------------------
int array::to_stream(teca_binary_stream &s) const
{
    // pack the metadata
    s.pack("array", 5);
    s.pack(this->name);
    s.pack(this->extent);

    // pack the size of the buffer
    size_t n_elem = this->internals->buffer->size();
    s.pack(n_elem);

    // always pack the data on the CPU
    std::shared_ptr<const double> d = this->internals->buffer->get_host_accessible();
    s.pack(d.get(), n_elem);

    return 0;
}

// --------------------------------------------------------------------------
int array::from_stream(teca_binary_stream &s)
{
    // unpack the metadata
    if (s.expect("array"))
    {
        TECA_ERROR("invalid stream")
        return -1;
    }
    s.unpack(this->name);
    s.unpack(this->extent);

    // unpack the buffer size
    size_t n_elem;
    s.unpack(n_elem);

    // always unpack the buffer on the CPU
    hamr::p_buffer<double> tmp =
        std::make_shared<hamr::buffer<double>>
            (teca_variant_array::allocator::malloc, n_elem);

    s.unpack(tmp->data(), n_elem);

    // move to the desired location
    this->internals->buffer->assign(ref_to(tmp));

    return 0;
}

// --------------------------------------------------------------------------
int array::to_stream(std::ostream &ostr) const
{
    // get the data on the CPU
    std::shared_ptr<const double> d = this->internals->buffer->get_host_accessible();


    ostr << "name=" << this->name
        << " extent=" << this->extent[0] << ", " << this->extent[1]
        << " values=";

    size_t n_elem = this->size();
    if (n_elem)
    {
        const double *pd = d.get();
        ostr << pd[0];
        for (size_t i = 1; i < n_elem; ++i)
            ostr << ", " << pd[i];
    }
    return 0;
}

// --------------------------------------------------------------------------
void array::debug_print() const
{
    this->internals->buffer->print();
}
