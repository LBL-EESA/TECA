#include "teca_cartesian_mesh.h"

#include <iostream>
using std::endl;

// --------------------------------------------------------------------------
teca_cartesian_mesh::teca_cartesian_mesh()
    : m_coordinate_arrays(teca_array_collection::New())
{}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::copy(dataset);
    m_coordinate_arrays->copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::shallow_copy(dataset);
    m_coordinate_arrays->shallow_copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::copy_metadata(const const_p_teca_dataset &dataset)
{
    const_p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::copy_metadata(dataset);

    m_coordinate_arrays->copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::swap(p_teca_dataset &dataset)
{
    p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::swap(dataset);
    m_coordinate_arrays->swap(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::to_stream(teca_binary_stream &s) const
{
    this->teca_mesh::to_stream(s);
    m_coordinate_arrays->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::from_stream(teca_binary_stream &s)
{
    this->teca_mesh::from_stream(s);
    m_coordinate_arrays->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::to_stream(std::ostream &s) const
{
    this->teca_mesh::to_stream(s);
    s << "coordinate arrays = {";
    m_coordinate_arrays->to_stream(s);
    s << "}" << endl;
}
