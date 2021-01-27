#include "teca_uniform_cartesian_mesh.h"
#include "teca_dataset_util.h"
#include "teca_bad_cast.h"

// --------------------------------------------------------------------------
teca_uniform_cartesian_mesh::teca_uniform_cartesian_mesh()
{}

// --------------------------------------------------------------------------
int teca_uniform_cartesian_mesh::get_type_code() const
{
    return teca_dataset_tt<teca_uniform_cartesian_mesh>::type_code;
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

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
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

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
        throw teca_bad_cast(safe_class_name(dataset), "teca_uniform_cartesian_mesh");

    if (this == other.get())
        return;

    this->teca_mesh::swap(dataset);
}
