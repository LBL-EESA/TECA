#include "array.h"
#include "teca_binary_stream.h"
#include "teca_bad_cast.h"

#include <hamr_buffer.h>

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
array::array(int alloc) : extent({0,0})
{
    this->internals = new array_internals;
    this->internals->buffer = hamr::buffer<double>::New(alloc);
}

// --------------------------------------------------------------------------
size_t array::size() const
{
    if (this->internals->buffer)
        return this->internals->buffer->size();

    return 0;
}

// --------------------------------------------------------------------------
p_array array::new_cpu_accessible()
{
    return array::New(hamr::buffer<double>::malloc);
}

// --------------------------------------------------------------------------
p_array array::new_cuda_accessible()
{
    return array::New(hamr::buffer<double>::cuda);
}

// --------------------------------------------------------------------------
p_array array::New(int alloc)
{
    return p_array(new array(alloc));
}

/*
// --------------------------------------------------------------------------
bool array::cpu_accessible() const
{
    return this->memory_resource->cpu_accessible();
}

// --------------------------------------------------------------------------
bool array::cuda_accessible() const
{
    return this->memory_resource->cuda_accessible();
}
*/

// --------------------------------------------------------------------------
p_teca_dataset array::new_copy() const
{
    p_teca_dataset a(new array());
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
std::shared_ptr<double> array::get_cpu_accessible()
{
    return this->internals->buffer->get_cpu_accessible();
}

// --------------------------------------------------------------------------
std::shared_ptr<const double> array::get_cpu_accessible() const
{
    return this->internals->buffer->get_cpu_accessible();
}


// --------------------------------------------------------------------------
std::shared_ptr<double> array::get_cuda_accessible()
{
    return this->internals->buffer->get_cuda_accessible();
}

// --------------------------------------------------------------------------
std::shared_ptr<const double> array::get_cuda_accessible() const
{
    return this->internals->buffer->get_cuda_accessible();
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
void array::copy(const const_p_teca_dataset &other)
{
    const_p_array other_a = std::dynamic_pointer_cast<const array>(other);

    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    if (this == other_a.get())
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;

    hamr::const_p_buffer<double> tmp = other_a->internals->buffer;
    this->internals->buffer->assign(tmp);
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
    std::shared_ptr<double> d = this->internals->buffer->get_cpu_accessible();
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
        hamr::buffer<double>::New(hamr::buffer<double>::malloc, n_elem);

    std::shared_ptr<double> pTmp = tmp->get_cpu_accessible();

    s.unpack(pTmp.get(), n_elem);

    // move to the desired location
    hamr::const_p_buffer<double> ctmp = tmp;
    this->internals->buffer->assign(ctmp);

    return 0;
}

// --------------------------------------------------------------------------
int array::to_stream(std::ostream &ostr) const
{
    // get the data on the CPU
    std::shared_ptr<const double> d = this->internals->buffer->get_cpu_accessible();


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
