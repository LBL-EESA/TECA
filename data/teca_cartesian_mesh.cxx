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
unsigned long teca_cartesian_mesh::get_number_of_points() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1);
}

// --------------------------------------------------------------------------
unsigned long teca_cartesian_mesh::get_number_of_cells() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0]) * (ext[3] - ext[2]) * (ext[5] - ext[4]);
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::get_type_code() const
{
    return teca_dataset_tt<teca_cartesian_mesh>::type_code;
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::copy(const const_p_teca_dataset &dataset,
    allocator alloc)
{
    this->teca_mesh::copy(dataset, alloc);

    const_p_teca_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(dataset);

    if ((!other) || (this == other.get()))
        return;

    m_coordinate_arrays->copy(other->m_coordinate_arrays, alloc);
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
void teca_cartesian_mesh::swap(const p_teca_dataset &dataset)
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
void teca_cartesian_mesh::update_x_coordinates(const p_teca_variant_array &array)
{
    m_coordinate_arrays->set("x", array);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::update_y_coordinates(const p_teca_variant_array &array)
{
    m_coordinate_arrays->set("y", array);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh::update_z_coordinates(const p_teca_variant_array &array)
{
    m_coordinate_arrays->set("z", array);
}

// --------------------------------------------------------------------------
int teca_cartesian_mesh::get_array_extent(const std::string &array_name,
    unsigned long array_extent[8]) const
{
    unsigned long mesh_extent[8] = {0};
    if (this->get_extent(mesh_extent) || this->get_temporal_extent(mesh_extent+6))
    {
        TECA_ERROR("Cartesian mesh dataset metadata issue. extent, "
            "and temporal_extent are required to get the extent of \""
            << array_name << "\"")
        return -1;
    }

    teca_metadata atts;
    teca_metadata array_atts;
    if (this->get_attributes(atts) || atts.get(array_name, array_atts))
    {
        // these are set during the execute phase and currently only
        // the cf reader does this correctly.
        /*TECA_WARNING("Cartesian mesh dataset metadata issue. array attributes "
            "are required to get the active extents of \"" << array_name
            << "\" the mesh extent will be used.")*/

        memcpy(array_extent, mesh_extent, 8*sizeof(unsigned long));

        return 1;
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
int teca_cartesian_mesh::get_array_shape(const std::string &array_name,
    unsigned long array_shape[4]) const
{
    unsigned long array_extent[8] = {0};
    if (this->get_array_extent(array_name, array_extent) < 0)
    {
        TECA_ERROR("Failed to get the shape of \"" << array_name << "\"")
        return -1;
    }

    array_shape[0] = array_extent[1] - array_extent[0] + 1;
    array_shape[1] = array_extent[3] - array_extent[2] + 1;
    array_shape[2] = array_extent[5] - array_extent[4] + 1;
    array_shape[3] = array_extent[7] - array_extent[6] + 1;

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
