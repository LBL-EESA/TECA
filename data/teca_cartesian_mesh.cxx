#include "teca_cartesian_mesh.h"
#include "teca_dataset_util.h"
#include "teca_bad_cast.h"
#include "teca_metadata.h"
#include "teca_metadata_util.h"

#include <iostream>

// --------------------------------------------------------------------------
teca_cartesian_mesh::teca_cartesian_mesh()
    : m_coordinate_arrays(teca_array_collection::New())
{}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::get_type_code() const
{
    return teca_dataset_tt<teca_cartesian_mesh>::type_code;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::copy(const const_p_teca_dataset &dataset)
{
    this->teca_mesh::copy(dataset);

    const_p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    this->teca_mesh::shallow_copy(dataset);

    p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->shallow_copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::copy_metadata(const const_p_teca_dataset &dataset)
{
    this->teca_mesh::copy_metadata(dataset);

    const_p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::swap(p_teca_dataset &dataset)
{
    this->teca_mesh::swap(dataset);

    p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_cartesian_mesh");

    if (this == other.get())
        return;

    m_coordinate_arrays->swap(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::set_x_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_x_coordinate_variable(var);
    m_coordinate_arrays->set("x", array);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::set_y_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_y_coordinate_variable(var);
    m_coordinate_arrays->set("y", array);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::set_z_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_z_coordinate_variable(var);
    m_coordinate_arrays->set("z", array);
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::get_array_extent(const std::string &array_name,
    unsigned long array_extent[6])
{
    teca_metadata atts;
    teca_metadata array_atts;
    unsigned long mesh_extent[6] = {0};
    if (this->get_extent(mesh_extent) || this->get_attributes(atts) ||
        atts.get(array_name, array_atts))
    {
        TECA_ERROR("Cartesian mesh dataset metadata issue. extent,"
            "attributes, and array attributes for \""
            << array_name << "\" are required")
        return -1;
    }

    if (teca_metadata_util::get_array_extent(array_atts,
        mesh_extent, array_extent))
    {
        // not necessarily an error
        return 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::to_stream(teca_binary_stream &s) const
{
    if (this->teca_mesh::to_stream(s)
        || m_coordinate_arrays->to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::from_stream(teca_binary_stream &s)
{
    if (this->teca_mesh::from_stream(s)
        || m_coordinate_arrays->from_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::to_stream(std::ostream &s) const
{
    this->teca_mesh::to_stream(s);
    s << "coordinate arrays = {";
    m_coordinate_arrays->to_stream(s);
    s << "}" << std::endl;
    return 0;
}
