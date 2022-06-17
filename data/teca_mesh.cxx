#include "teca_mesh.h"
#include "teca_array_attributes.h"
#include "teca_bad_cast.h"

teca_mesh::impl_t::impl_t()
{
    this->point_arrays = teca_array_collection::New();
    this->cell_arrays = teca_array_collection::New();
    this->x_edge_arrays = teca_array_collection::New();
    this->y_edge_arrays = teca_array_collection::New();
    this->z_edge_arrays = teca_array_collection::New();
    this->x_face_arrays = teca_array_collection::New();
    this->y_face_arrays = teca_array_collection::New();
    this->z_face_arrays = teca_array_collection::New();
    this->info_arrays = teca_array_collection::New();
}

// --------------------------------------------------------------------------
teca_mesh::teca_mesh()
    : m_impl(std::make_shared<teca_mesh::impl_t>())
{}

// --------------------------------------------------------------------------
void teca_mesh::copy(const const_p_teca_dataset &dataset, allocator alloc)
{
    const_p_teca_mesh other
        = std::dynamic_pointer_cast<const teca_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_mesh");

    if (this == other.get())
        return;

    this->teca_dataset::copy(dataset, alloc);

    m_impl = std::make_shared<teca_mesh::impl_t>();
    m_impl->point_arrays->copy(other->m_impl->point_arrays, alloc);
    m_impl->cell_arrays->copy(other->m_impl->cell_arrays, alloc);
    m_impl->x_edge_arrays->copy(other->m_impl->x_edge_arrays, alloc);
    m_impl->y_edge_arrays->copy(other->m_impl->y_edge_arrays, alloc);
    m_impl->z_edge_arrays->copy(other->m_impl->z_edge_arrays, alloc);
    m_impl->x_face_arrays->copy(other->m_impl->x_face_arrays, alloc);
    m_impl->y_face_arrays->copy(other->m_impl->y_face_arrays, alloc);
    m_impl->z_face_arrays->copy(other->m_impl->z_face_arrays, alloc);
    m_impl->info_arrays->copy(other->m_impl->info_arrays, alloc);
}

// --------------------------------------------------------------------------
void teca_mesh::shallow_copy(const p_teca_dataset &dataset)
{
    p_teca_mesh other
        = std::dynamic_pointer_cast<teca_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_mesh");

    if (this == other.get())
        return;

    this->teca_dataset::shallow_copy(dataset);

    m_impl = std::make_shared<teca_mesh::impl_t>();
    m_impl->point_arrays->shallow_copy(other->m_impl->point_arrays);
    m_impl->cell_arrays->shallow_copy(other->m_impl->cell_arrays);
    m_impl->x_edge_arrays->shallow_copy(other->m_impl->x_edge_arrays);
    m_impl->y_edge_arrays->shallow_copy(other->m_impl->y_edge_arrays);
    m_impl->z_edge_arrays->shallow_copy(other->m_impl->z_edge_arrays);
    m_impl->x_face_arrays->shallow_copy(other->m_impl->x_face_arrays);
    m_impl->y_face_arrays->shallow_copy(other->m_impl->y_face_arrays);
    m_impl->z_face_arrays->shallow_copy(other->m_impl->z_face_arrays);
    m_impl->info_arrays->shallow_copy(other->m_impl->info_arrays);
}

// --------------------------------------------------------------------------
void teca_mesh::append_arrays(const const_p_teca_mesh &other)
{
    if (this == other.get())
        return;

    m_impl->point_arrays->append(other->m_impl->point_arrays);
    m_impl->cell_arrays->append(other->m_impl->cell_arrays);
    m_impl->x_edge_arrays->append(other->m_impl->x_edge_arrays);
    m_impl->y_edge_arrays->append(other->m_impl->y_edge_arrays);
    m_impl->z_edge_arrays->append(other->m_impl->z_edge_arrays);
    m_impl->x_face_arrays->append(other->m_impl->x_face_arrays);
    m_impl->y_face_arrays->append(other->m_impl->y_face_arrays);
    m_impl->z_face_arrays->append(other->m_impl->z_face_arrays);
    m_impl->info_arrays->append(other->m_impl->info_arrays);
}

// --------------------------------------------------------------------------
void teca_mesh::shallow_append_arrays(const p_teca_mesh &other)
{
    if (this == other.get())
        return;

    m_impl->point_arrays->shallow_append(other->m_impl->point_arrays);
    m_impl->cell_arrays->shallow_append(other->m_impl->cell_arrays);
    m_impl->x_edge_arrays->shallow_append(other->m_impl->x_edge_arrays);
    m_impl->y_edge_arrays->shallow_append(other->m_impl->y_edge_arrays);
    m_impl->z_edge_arrays->shallow_append(other->m_impl->z_edge_arrays);
    m_impl->x_face_arrays->shallow_append(other->m_impl->x_face_arrays);
    m_impl->y_face_arrays->shallow_append(other->m_impl->y_face_arrays);
    m_impl->z_face_arrays->shallow_append(other->m_impl->z_face_arrays);
    m_impl->info_arrays->shallow_append(other->m_impl->info_arrays);
}

// --------------------------------------------------------------------------
void teca_mesh::swap(const p_teca_dataset &dataset)
{
    p_teca_mesh other
        = std::dynamic_pointer_cast<teca_mesh>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_mesh");

    if (this == other.get())
        return;

    this->teca_dataset::swap(dataset);

    std::swap(m_impl, other->m_impl);
}

// --------------------------------------------------------------------------
p_teca_array_collection &teca_mesh::get_arrays(int centering)
{
    switch (centering)
    {
        case teca_array_attributes::invalid_value:
            TECA_ERROR("detected invalid_value in centering")
            break;
        case teca_array_attributes::cell_centering:
            return m_impl->cell_arrays;
            break;
        case teca_array_attributes::x_face_centering:
            return m_impl->x_face_arrays;
            break;
        case teca_array_attributes::y_face_centering:
            return m_impl->y_face_arrays;
            break;
        case teca_array_attributes::z_face_centering:
            return m_impl->z_face_arrays;
            break;
        case teca_array_attributes::x_edge_centering:
            return m_impl->x_edge_arrays;
            break;
        case teca_array_attributes::y_edge_centering:
            return m_impl->y_edge_arrays;
            break;
        case teca_array_attributes::z_edge_centering:
            return m_impl->z_edge_arrays;
            break;
        case teca_array_attributes::point_centering:
            return m_impl->point_arrays;
            break;
        case teca_array_attributes::no_centering:
            return m_impl->info_arrays;
            break;
        default:
            TECA_ERROR("this centering is undefined " << centering)
    }

    // because there is no such thing as a null reference
    return m_impl->invalid;
}

// --------------------------------------------------------------------------
const_p_teca_array_collection teca_mesh::get_arrays(int centering) const
{
    return const_cast<teca_mesh*>(this)->get_arrays(centering);
}

// --------------------------------------------------------------------------
int teca_mesh::to_stream(teca_binary_stream &s) const
{
    if (this->teca_dataset::to_stream(s)
        || m_impl->point_arrays->to_stream(s)
        || m_impl->cell_arrays->to_stream(s)
        || m_impl->x_edge_arrays->to_stream(s)
        || m_impl->y_edge_arrays->to_stream(s)
        || m_impl->z_edge_arrays->to_stream(s)
        || m_impl->x_face_arrays->to_stream(s)
        || m_impl->y_face_arrays->to_stream(s)
        || m_impl->z_face_arrays->to_stream(s)
        || m_impl->info_arrays->to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_mesh::from_stream(teca_binary_stream &s)
{
    if (this->teca_dataset::from_stream(s)
        || m_impl->point_arrays->from_stream(s)
        || m_impl->cell_arrays->from_stream(s)
        || m_impl->x_edge_arrays->from_stream(s)
        || m_impl->y_edge_arrays->from_stream(s)
        || m_impl->z_edge_arrays->from_stream(s)
        || m_impl->x_face_arrays->from_stream(s)
        || m_impl->y_face_arrays->from_stream(s)
        || m_impl->z_face_arrays->from_stream(s)
        || m_impl->info_arrays->from_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_mesh::to_stream(std::ostream &s) const
{
    this->teca_dataset::to_stream(s);

    s << "point arrays = ";
    m_impl->point_arrays->to_stream(s);
    s << std::endl;
    s << "cell arrays = ";
    m_impl->cell_arrays->to_stream(s);
    s << std::endl;

    return 0;
}

// --------------------------------------------------------------------------
bool teca_mesh::empty() const noexcept
{
    return
        !( m_impl->point_arrays->size()
        || m_impl->cell_arrays->size()
        || m_impl->x_edge_arrays->size()
        || m_impl->y_edge_arrays->size()
        || m_impl->z_edge_arrays->size()
        || m_impl->x_face_arrays->size()
        || m_impl->y_face_arrays->size()
        || m_impl->z_face_arrays->size()
        || m_impl->info_arrays->size());
}

// --------------------------------------------------------------------------
void teca_mesh::set_time_step(const unsigned long & val)
{
    unsigned long extent[2] = {val, val};
    this->get_metadata().set<unsigned long>("temporal_extent", extent);
}

// --------------------------------------------------------------------------
int teca_mesh::get_time_step(unsigned long &val) const
{
    unsigned long extent[2] = {0ul};

    int ierr = this->get_metadata().get<unsigned long>(
        "temporal_extent", extent);

    val = extent[0];

    return ierr;
}

// --------------------------------------------------------------------------
void teca_mesh::set_time(const double & val)
{
    double bounds[2] = {val, val};
    this->get_metadata().set<double>("temporal_bounds", bounds);
}

// --------------------------------------------------------------------------
int teca_mesh::get_time(double &val) const
{
    double bounds[2] = {double()};

    int ierr = this->get_metadata().get<double>(
        "temporal_bounds", bounds);

    val = bounds[0];

    return ierr;
}
