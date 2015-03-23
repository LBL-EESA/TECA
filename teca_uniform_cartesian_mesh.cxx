#include "teca_uniform_cartesian_mesh.h"

// --------------------------------------------------------------------------
teca_uniform_cartesian_mesh::teca_uniform_cartesian_mesh()
    : m_impl(std::make_shared<teca_uniform_cartesian_mesh::impl_t>())
{
}

// --------------------------------------------------------------------------
p_teca_dataset teca_uniform_cartesian_mesh::new_copy() const
{
    p_teca_uniform_cartesian_mesh m = teca_uniform_cartesian_mesh::New();
    m->copy(this->shared_from_this());
    return m;
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    m_impl = std::make_shared<teca_uniform_cartesian_mesh::impl_t>();
    m_impl->metadata = other->m_impl->metadata;
    m_impl->point_arrays->copy(other->m_impl->point_arrays);
    m_impl->cell_arrays->copy(other->m_impl->cell_arrays);
    m_impl->edge_arrays->copy(other->m_impl->edge_arrays);
    m_impl->face_arrays->copy(other->m_impl->face_arrays);
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

    m_impl = std::make_shared<teca_uniform_cartesian_mesh::impl_t>();

    m_impl->metadata = other->m_impl->metadata;
    m_impl->point_arrays->shallow_copy(other->m_impl->point_arrays);
    m_impl->cell_arrays->shallow_copy(other->m_impl->cell_arrays);
    m_impl->edge_arrays->shallow_copy(other->m_impl->edge_arrays);
    m_impl->face_arrays->shallow_copy(other->m_impl->face_arrays);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::copy_metadata(const const_p_teca_dataset &dataset)
{
    const_p_teca_uniform_cartesian_mesh other
        = std::dynamic_pointer_cast<const teca_uniform_cartesian_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    m_impl->metadata = other->m_impl->metadata;
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

    std::shared_ptr<teca_uniform_cartesian_mesh::impl_t> tmp = m_impl;
    m_impl = other->m_impl;
    other->m_impl = tmp;
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::to_stream(teca_binary_stream &s) const
{
    m_impl->metadata.to_stream(s);
    m_impl->point_arrays->to_stream(s);
    m_impl->cell_arrays->to_stream(s);
    m_impl->edge_arrays->to_stream(s);
    m_impl->face_arrays->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::from_stream(teca_binary_stream &s)
{
    m_impl->metadata.from_stream(s);
    m_impl->point_arrays->from_stream(s);
    m_impl->cell_arrays->from_stream(s);
    m_impl->edge_arrays->from_stream(s);
    m_impl->face_arrays->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_uniform_cartesian_mesh::to_stream(std::ostream &) const
{
}
