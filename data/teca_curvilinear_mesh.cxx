#include "teca_curvilinear_mesh.h"
#include "teca_dataset_util.h"
#include "teca_bad_cast.h"

#include <iostream>
using std::endl;

// --------------------------------------------------------------------------
teca_curvilinear_mesh::teca_curvilinear_mesh()
    : m_coordinate_arrays(teca_array_collection::New())
{}

// --------------------------------------------------------------------------
unsigned long teca_curvilinear_mesh::get_number_of_points() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1);
}

// --------------------------------------------------------------------------
unsigned long teca_curvilinear_mesh::get_number_of_cells() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0]) * (ext[3] - ext[2]) * (ext[5] - ext[4]);
}

// --------------------------------------------------------------------------
int teca_curvilinear_mesh::get_type_code() const
{
    return teca_dataset_tt<teca_curvilinear_mesh>::type_code;
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::copy(const const_p_teca_dataset &dataset,
    allocator alloc)
{
    this->teca_mesh::copy(dataset, alloc);

    const_p_teca_curvilinear_mesh other
        = std::dynamic_pointer_cast<const teca_curvilinear_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->copy(other->m_coordinate_arrays, alloc);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    this->teca_mesh::shallow_copy(dataset);

    p_teca_curvilinear_mesh other
        = std::dynamic_pointer_cast<teca_curvilinear_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->shallow_copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::copy_metadata(const const_p_teca_dataset &dataset)
{
    this->teca_mesh::copy_metadata(dataset);

    const_p_teca_curvilinear_mesh other
        = std::dynamic_pointer_cast<const teca_curvilinear_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->copy(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::swap(const p_teca_dataset &dataset)
{
    this->teca_mesh::swap(dataset);

    p_teca_curvilinear_mesh other
        = std::dynamic_pointer_cast<teca_curvilinear_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_curvilinear_mesh");

    if (this == other.get())
        return;

    m_coordinate_arrays->swap(other->m_coordinate_arrays);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::set_x_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_x_coordinate_variable(var);
    m_coordinate_arrays->set("x", array);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::set_y_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_y_coordinate_variable(var);
    m_coordinate_arrays->set("y", array);
}

// --------------------------------------------------------------------------
void teca_curvilinear_mesh::set_z_coordinates(const std::string &var,
    const p_teca_variant_array &array)
{
    this->set_z_coordinate_variable(var);
    m_coordinate_arrays->set("z", array);
}

// --------------------------------------------------------------------------
int teca_curvilinear_mesh::to_stream(teca_binary_stream &s) const
{
    if (this->teca_mesh::to_stream(s)
        || m_coordinate_arrays->to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_curvilinear_mesh::from_stream(teca_binary_stream &s)
{
    if (this->teca_mesh::from_stream(s)
        || m_coordinate_arrays->from_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_curvilinear_mesh::to_stream(std::ostream &s) const
{
    this->teca_mesh::to_stream(s);
    s << "coordinate arrays = {";
    m_coordinate_arrays->to_stream(s);
    s << "}" << endl;
    return 0;
}
