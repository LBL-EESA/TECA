#include "teca_arakawa_c_grid.h"
#include "teca_dataset_util.h"
#include "teca_bad_cast.h"

#include <iostream>
#include <cctype>

teca_arakawa_c_grid::impl_t::impl_t()
{
    m_x_coordinates = nullptr;
    m_y_coordinates = nullptr;
    u_x_coordinates = nullptr;
    u_y_coordinates = nullptr;
    v_x_coordinates = nullptr;
    v_y_coordinates = nullptr;
    m_z_coordinates = nullptr;
    w_z_coordinates = nullptr;
    t_coordinates = nullptr;
}

// --------------------------------------------------------------------------
teca_arakawa_c_grid::teca_arakawa_c_grid()
    : m_impl(std::make_shared<teca_arakawa_c_grid::impl_t>())
{}

// --------------------------------------------------------------------------
unsigned long teca_arakawa_c_grid::get_number_of_points() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1) * (ext[5] - ext[4] + 1);
}

// --------------------------------------------------------------------------
unsigned long teca_arakawa_c_grid::get_number_of_cells() const
{
    unsigned long ext[6];
    this->get_extent(ext);

    return (ext[1] - ext[0]) * (ext[3] - ext[2]) * (ext[5] - ext[4]);
}

// --------------------------------------------------------------------------
int teca_arakawa_c_grid::get_type_code() const
{
    return teca_dataset_tt<teca_arakawa_c_grid>::type_code;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_m_x_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_m_x_coordinate_variable(name);
    m_impl->m_x_coordinates = a;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_m_y_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_m_y_coordinate_variable(name);
    m_impl->m_y_coordinates = a;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_m_x_coordinates()
{
    return m_impl->m_x_coordinates;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_m_y_coordinates()
{
    return m_impl->m_y_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_m_x_coordinates() const
{
    return m_impl->m_x_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_m_y_coordinates() const
{
    return m_impl->m_y_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_u_x_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_u_x_coordinate_variable(name);
    m_impl->u_x_coordinates = a;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_u_y_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_u_y_coordinate_variable(name);
    m_impl->u_y_coordinates = a;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_u_x_coordinates()
{
    return m_impl->u_x_coordinates;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_u_y_coordinates()
{
    return m_impl->u_y_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_u_x_coordinates() const
{
    return m_impl->u_x_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_u_y_coordinates() const
{
    return m_impl->u_y_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_v_x_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_v_x_coordinate_variable(name);
    m_impl->v_x_coordinates = a;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_v_y_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_v_y_coordinate_variable(name);
    m_impl->v_y_coordinates = a;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_v_x_coordinates()
{
    return m_impl->v_x_coordinates;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_v_y_coordinates()
{
    return m_impl->v_y_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_v_x_coordinates() const
{
    return m_impl->v_x_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_v_y_coordinates() const
{
    return m_impl->v_y_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_m_z_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_m_z_coordinate_variable(name);
    m_impl->m_z_coordinates = a;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_m_z_coordinates()
{
    return m_impl->m_z_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_m_z_coordinates() const
{
    return m_impl->m_z_coordinates;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_w_z_coordinates()
{
    return m_impl->w_z_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_w_z_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_w_z_coordinate_variable(name);
    m_impl->w_z_coordinates = a;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_w_z_coordinates() const
{
    return m_impl->w_z_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::set_t_coordinates(const std::string &name,
    const p_teca_variant_array &a)
{
    this->set_t_coordinate_variable(name);
    m_impl->t_coordinates = a;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_arakawa_c_grid::get_t_coordinates()
{
    return m_impl->t_coordinates;
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_arakawa_c_grid::get_t_coordinates() const
{
    return m_impl->t_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::copy(const const_p_teca_dataset &dataset,
    allocator alloc)
{
    const_p_teca_arakawa_c_grid other
        = std::dynamic_pointer_cast<const teca_arakawa_c_grid>(dataset);

    if (!other || (this == other.get()))
        return;

    this->teca_mesh::copy(dataset, alloc);

    m_impl = std::make_shared<teca_arakawa_c_grid::impl_t>();
    m_impl->m_x_coordinates = other->m_impl->m_x_coordinates->new_copy(alloc);
    m_impl->m_y_coordinates = other->m_impl->m_y_coordinates->new_copy(alloc);
    m_impl->u_x_coordinates = other->m_impl->u_x_coordinates->new_copy(alloc);
    m_impl->u_y_coordinates = other->m_impl->u_y_coordinates->new_copy(alloc);
    m_impl->v_x_coordinates = other->m_impl->v_x_coordinates->new_copy(alloc);
    m_impl->v_y_coordinates = other->m_impl->v_y_coordinates->new_copy(alloc);
    m_impl->m_z_coordinates = other->m_impl->m_z_coordinates->new_copy(alloc);
    m_impl->w_z_coordinates = other->m_impl->w_z_coordinates->new_copy(alloc);
    m_impl->t_coordinates = other->m_impl->t_coordinates->new_copy(alloc);
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::shallow_copy(const p_teca_dataset &dataset)
{
    const_p_teca_arakawa_c_grid other
        = std::dynamic_pointer_cast<const teca_arakawa_c_grid>(dataset);

    if (!other || (this == other.get()))
        return;

    this->teca_mesh::shallow_copy(dataset);

    m_impl = std::make_shared<teca_arakawa_c_grid::impl_t>();
    m_impl->m_x_coordinates = other->m_impl->m_x_coordinates;
    m_impl->m_y_coordinates = other->m_impl->m_y_coordinates;
    m_impl->u_x_coordinates = other->m_impl->u_x_coordinates;
    m_impl->u_y_coordinates = other->m_impl->u_y_coordinates;
    m_impl->v_x_coordinates = other->m_impl->v_x_coordinates;
    m_impl->v_y_coordinates = other->m_impl->v_y_coordinates;
    m_impl->m_z_coordinates = other->m_impl->m_z_coordinates;
    m_impl->w_z_coordinates = other->m_impl->w_z_coordinates;
    m_impl->t_coordinates = other->m_impl->t_coordinates;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::copy_metadata(const const_p_teca_dataset &dataset)
{
    this->teca_mesh::copy_metadata(dataset);

    const_p_teca_arakawa_c_grid other
        = std::dynamic_pointer_cast<const teca_arakawa_c_grid>(dataset);

    if ((!other) || (this == other.get()))
        return;
}

// --------------------------------------------------------------------------
void teca_arakawa_c_grid::swap(const p_teca_dataset &dataset)
{
    this->teca_mesh::swap(dataset);

    p_teca_arakawa_c_grid other
        = std::dynamic_pointer_cast<teca_arakawa_c_grid>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_arakawa_c_grid");

    if (this == other.get())
        return;

    std::swap(m_impl, other->m_impl);
}

// --------------------------------------------------------------------------
int teca_arakawa_c_grid::to_stream(teca_binary_stream &s) const
{
    if (this->teca_mesh::to_stream(s)
        || m_impl->m_x_coordinates->to_stream(s)
        || m_impl->m_y_coordinates->to_stream(s)
        || m_impl->u_x_coordinates->to_stream(s)
        || m_impl->u_y_coordinates->to_stream(s)
        || m_impl->v_x_coordinates->to_stream(s)
        || m_impl->v_y_coordinates->to_stream(s)
        || m_impl->m_z_coordinates->to_stream(s)
        || m_impl->w_z_coordinates->to_stream(s)
        || m_impl->t_coordinates->to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_arakawa_c_grid::from_stream(teca_binary_stream &s)
{
    if (this->teca_mesh::from_stream(s)
        || m_impl->m_x_coordinates->from_stream(s)
        || m_impl->m_y_coordinates->from_stream(s)
        || m_impl->u_x_coordinates->from_stream(s)
        || m_impl->u_y_coordinates->from_stream(s)
        || m_impl->v_x_coordinates->from_stream(s)
        || m_impl->v_y_coordinates->from_stream(s)
        || m_impl->m_z_coordinates->from_stream(s)
        || m_impl->w_z_coordinates->from_stream(s)
        || m_impl->t_coordinates->from_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_arakawa_c_grid::to_stream(std::ostream &s) const
{
    if (this->teca_mesh::to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
bool teca_arakawa_c_grid::empty() const noexcept
{

    return teca_mesh::empty() &&
        !( m_impl->m_x_coordinates->size()
        || m_impl->m_y_coordinates->size()
        || m_impl->u_x_coordinates->size()
        || m_impl->u_y_coordinates->size()
        || m_impl->v_x_coordinates->size()
        || m_impl->v_y_coordinates->size()
        || m_impl->m_z_coordinates->size()
        || m_impl->w_z_coordinates->size()
        || m_impl->t_coordinates->size());
}

