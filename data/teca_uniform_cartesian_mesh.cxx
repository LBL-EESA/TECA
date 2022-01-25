#include "teca_uniform_cartesian_mesh.h"
#include "teca_dataset_util.h"
#include "teca_bad_cast.h"

// --------------------------------------------------------------------------
teca_uniform_cartesian_mesh::teca_uniform_cartesian_mesh()
{}

// --------------------------------------------------------------------------
unsigned long teca_uniform_cartesian_mesh::get_number_of_points() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1);
}

// --------------------------------------------------------------------------
unsigned long teca_uniform_cartesian_mesh::get_number_of_cells() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0]) * (ext[3] - ext[2]) * (ext[5] - ext[4]);
}

// --------------------------------------------------------------------------
int teca_uniform_cartesian_mesh::get_type_code() const
{
    return teca_dataset_tt<teca_uniform_cartesian_mesh>::type_code;
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::copy(const const_p_teca_dataset &dataset,
    allocator alloc)
{
    const_p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

    if (this == other.get())
        return;

    this->teca_mesh::copy(dataset, alloc);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

    if (this == other.get())
        return;

    this->teca_mesh::shallow_copy(dataset);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::swap(const p_teca_dataset &dataset)
{
    p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

    if (this == other.get())
        return;

    this->teca_mesh::swap(dataset);
}
