#include "array.h"
#include "teca_binary_stream.h"
#include "teca_bad_cast.h"

#include <hamm_cuda_mm_memory_resource.h>
#include <hamm_cpu_memory_resource.h>

#include <utility>
#include <iostream>
#include <sstream>

using std::ostringstream;
using std::ostream;

// --------------------------------------------------------------------------
array::array() : extent({0,0}), memory_resource{nullptr}, data{nullptr}
{
    this->memory_resource = hamm_cpu_memory_resource::New();
    this->data = std::make_shared<hamm_pmr_vector<double>>(this->memory_resource.get());
}

// --------------------------------------------------------------------------
array::array(const p_hamm_memory_resource &alloc) :
    extent({0,0}), memory_resource{alloc}, data{nullptr}
{
    this->data = std::make_shared<hamm_pmr_vector<double>>(this->memory_resource.get());

    /*std::cerr << "Created " << this->get_class_name() << " with an "
        << alloc->get_class_name() << " memory_resource" << std::endl;*/
}

// --------------------------------------------------------------------------
p_array array::new_cpu_accessible()
{
    return array::New(hamm_cpu_memory_resource::New());
}

// --------------------------------------------------------------------------
p_array array::new_cuda_accessible()
{
    return array::New(hamm_cuda_mm_memory_resource::New());
}

// --------------------------------------------------------------------------
p_array array::New(const p_hamm_memory_resource &alloc)
{
    return p_array(new array(alloc));
}

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
void array::resize(size_t n)
{
    this->data->resize(n, 0.0);
    this->extent = {0, n};
}

// --------------------------------------------------------------------------
void array::clear()
{
    this->data->clear();
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

    // TODO - should we copy the memory_resource type as well? if so would need
    // to add API for transfering from device to device.
    /*this->memory_resource = other_a->memory_resource->new_instance();
    this->data = std::make_shared<hamm_pmr_vector<double>>(this->memory_resource.get());*/

    this->data->assign(other_a->data->begin(), other_a->data->end());
}

// --------------------------------------------------------------------------
void array::shallow_copy(const p_teca_dataset &other)
{
    // TODO : need to store internal data in shared ptr
    // to support shallow copies.
    this->copy(other);
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
    this->data->resize(this->extent[1]-this->extent[0]);
}

// --------------------------------------------------------------------------
void array::swap(const p_teca_dataset &other)
{
    p_array other_a = std::dynamic_pointer_cast<array>(other);
    if (!other_a)
        throw teca_bad_cast(safe_class_name(other), "array");

    std::swap(this->name, other_a->name);
    std::swap(this->extent, other_a->extent);
    std::swap(this->memory_resource, other_a->memory_resource);
    std::swap(this->data, other_a->data);
}

// --------------------------------------------------------------------------
int array::to_stream(teca_binary_stream &s) const
{
    s.pack("array", 5);
    s.pack(this->name);
    s.pack(this->extent);
    s.pack(this->data);
    return 0;
}

// --------------------------------------------------------------------------
int array::from_stream(teca_binary_stream &s)
{
    if (s.expect("array"))
    {
        TECA_ERROR("invalid stream")
        return -1;
    }
    s.unpack(this->name);
    s.unpack(this->extent);
    s.unpack(this->data);
    return 0;
}

// --------------------------------------------------------------------------
int array::to_stream(std::ostream &ostr) const
{
    ostr << "name=" << this->name
        << " extent=" << this->extent[0] << ", " << this->extent[1]
        << " memory_resource=" << this->memory_resource->get_class_name()
        << " values=";

    size_t n_elem = this->size();
    if (n_elem)
    {
        ostr << this->data->at(0);
        for (size_t i = 1; i < n_elem; ++i)
            ostr << ", " << this->data->at(i);
    }
    return 0;
}
