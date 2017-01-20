#include "teca_uniform_cartesian_mesh.h"

// --------------------------------------------------------------------------
teca_uniform_cartesian_mesh::teca_uniform_cartesian_mesh()
{}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::copy(dataset);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::shallow_copy(dataset);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::swap(p_teca_dataset &dataset)
{
    p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->teca_mesh::swap(dataset);
}
