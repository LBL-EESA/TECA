#include "array.h"
#include "teca_binary_stream.h"
#include <utility>
#include <iostream>
#include <sstream>

using std::ostringstream;
using std::ostream;

// --------------------------------------------------------------------------
p_teca_dataset array::new_copy() const
{
    p_teca_dataset a(new array());
    a->copy(this);
    return a;
}

// --------------------------------------------------------------------------
void array::resize(size_t n)
{
    this->data.resize(n, 0.0);
    this->extent = {0, n};
}

// --------------------------------------------------------------------------
void array::clear()
{
    this->data.clear();
    this->extent[0] = 0;
    this->extent[1] = 0;
}

// --------------------------------------------------------------------------
void array::copy(const teca_dataset *other)
{
    const array *other_a = dynamic_cast<const array*>(other);
    if (! other_a)
        throw std::bad_cast();

    if (this == other_a)
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;
    this->data = other_a->data;
}

// --------------------------------------------------------------------------
void array::shallow_copy(const teca_dataset *other)
{
    // TODO : need to store internal data in shared ptr
    // to support shallow copies.
    this->copy(other);
}

// --------------------------------------------------------------------------
void array::copy_metadata(const teca_dataset *other)
{
    const array *other_a = dynamic_cast<const array*>(other);
    if (! other_a)
        throw std::bad_cast();

    if (this == other_a)
        return;

    this->name = other_a->name;
    this->extent = other_a->extent;
    this->data.resize(this->extent[1]-this->extent[0]);
}

// --------------------------------------------------------------------------
void array::swap(teca_dataset *other)
{
    array *other_a = dynamic_cast<array*>(other);
    if (! other_a)
        throw std::bad_cast();

    std::swap(this->name, other_a->name);
    std::swap(this->extent, other_a->extent);
    std::swap(this->data, other_a->data);
}

// --------------------------------------------------------------------------
void array::to_stream(teca_binary_stream &s) const
{
    s.pack(this->name);
    s.pack(this->extent);
    s.pack(this->data);
}

// --------------------------------------------------------------------------
void array::from_stream(teca_binary_stream &s)
{
    s.unpack(this->name);
    s.unpack(this->extent);
    s.unpack(this->data);
}

// --------------------------------------------------------------------------
void array::to_stream(std::ostream &ostr) const
{
    ostr << "name=" << this->name
        << " extent="
        << this->extent[0] << ", " << this->extent[1]
        << " values=";

    size_t n_elem = this->size();
    if (n_elem)
    {
        ostr << this->data[0];
        for (size_t i = 1; i < n_elem; ++i)
            ostr << ", " << this->data[i];
    }
}
