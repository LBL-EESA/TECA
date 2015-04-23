#include "teca_mesh.h"

// --------------------------------------------------------------------------
teca_mesh::teca_mesh()
    : m_impl(std::make_shared<teca_mesh::impl_t>())
{
    m_impl->point_arrays = teca_array_collection::New();
    m_impl->cell_arrays = teca_array_collection::New();
    m_impl->edge_arrays = teca_array_collection::New();
    m_impl->face_arrays = teca_array_collection::New();
    m_impl->info_arrays = teca_array_collection::New();
}

// --------------------------------------------------------------------------
void teca_mesh::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_mesh other
        = std::dynamic_pointer_cast<const teca_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    m_impl = std::make_shared<teca_mesh::impl_t>();

    m_impl->metadata = other->m_impl->metadata;
    m_impl->point_arrays->copy(other->m_impl->point_arrays);
    m_impl->cell_arrays->copy(other->m_impl->cell_arrays);
    m_impl->edge_arrays->copy(other->m_impl->edge_arrays);
    m_impl->face_arrays->copy(other->m_impl->face_arrays);
    m_impl->info_arrays->copy(other->m_impl->info_arrays);
}

// --------------------------------------------------------------------------
void teca_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    p_teca_mesh other
        = std::dynamic_pointer_cast<teca_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    m_impl = std::make_shared<teca_mesh::impl_t>();

    m_impl->metadata = other->m_impl->metadata;
    m_impl->point_arrays->shallow_copy(other->m_impl->point_arrays);
    m_impl->cell_arrays->shallow_copy(other->m_impl->cell_arrays);
    m_impl->edge_arrays->shallow_copy(other->m_impl->edge_arrays);
    m_impl->face_arrays->shallow_copy(other->m_impl->face_arrays);
    m_impl->info_arrays->shallow_copy(other->m_impl->info_arrays);
}

// --------------------------------------------------------------------------
void teca_mesh::swap(p_teca_dataset &dataset)
{
    p_teca_mesh other
        = std::dynamic_pointer_cast<teca_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    std::swap(m_impl, other->m_impl);
}

// --------------------------------------------------------------------------
void teca_mesh::copy_metadata(const const_p_teca_dataset &dataset)
{
    const_p_teca_mesh other
        = std::dynamic_pointer_cast<const teca_mesh>(dataset);

    if (!other)
        throw std::bad_cast();

    m_impl->metadata = other->m_impl->metadata;
}

// --------------------------------------------------------------------------
void teca_mesh::to_stream(teca_binary_stream &s) const
{
    m_impl->metadata.to_stream(s);
    m_impl->point_arrays->to_stream(s);
    m_impl->cell_arrays->to_stream(s);
    m_impl->edge_arrays->to_stream(s);
    m_impl->face_arrays->to_stream(s);
    m_impl->info_arrays->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_mesh::from_stream(teca_binary_stream &s)
{
    m_impl->metadata.from_stream(s);
    m_impl->point_arrays->from_stream(s);
    m_impl->cell_arrays->from_stream(s);
    m_impl->edge_arrays->from_stream(s);
    m_impl->face_arrays->from_stream(s);
    m_impl->info_arrays->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_mesh::to_stream(std::ostream &) const
{
    // TODO
}

// --------------------------------------------------------------------------
bool teca_mesh::empty() const TECA_NOEXCEPT
{
    return
        !( m_impl->point_arrays->size()
        || m_impl->cell_arrays->size()
        || m_impl->edge_arrays->size()
        || m_impl->face_arrays->size()
        || m_impl->info_arrays->size());
}
