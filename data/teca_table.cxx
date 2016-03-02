#include "teca_table.h"

#include "teca_binary_stream.h"

using std::vector;
using std::map;

teca_table::impl_t::impl_t() :
    columns(teca_array_collection::New()), active_column(0)
{}


// --------------------------------------------------------------------------
teca_table::teca_table() : m_impl(new teca_table::impl_t())
{}

// --------------------------------------------------------------------------
void teca_table::clear()
{
    m_impl->metadata.clear();
    m_impl->columns->clear();
    m_impl->active_column = 0;
}

// --------------------------------------------------------------------------
bool teca_table::empty() const noexcept
{
    return m_impl->columns->size() == 0;
}

// --------------------------------------------------------------------------
unsigned int teca_table::get_number_of_columns() const noexcept
{
    return m_impl->columns->size();
}

// --------------------------------------------------------------------------
unsigned long teca_table::get_number_of_rows() const noexcept
{
    if (m_impl->columns->size())
        return m_impl->columns->get(0)->size();

    return 0;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_table::get_column(const std::string &col_name)
{
    return m_impl->columns->get(col_name);
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_table::get_column(const std::string &col_name) const
{
    return m_impl->columns->get(col_name);
}

// --------------------------------------------------------------------------
void teca_table::resize(unsigned long n)
{
    unsigned int n_cols = m_impl->columns->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        m_impl->columns->get(i)->resize(n);
}

// --------------------------------------------------------------------------
void teca_table::reserve(unsigned long n)
{
    unsigned int n_cols = m_impl->columns->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        m_impl->columns->get(i)->reserve(n);
}

// --------------------------------------------------------------------------
void teca_table::to_stream(teca_binary_stream &s) const
{
    m_impl->metadata.to_stream(s);
    m_impl->columns->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_table::from_stream(teca_binary_stream &s)
{
    this->clear();
    m_impl->metadata.from_stream(s);
    m_impl->columns->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_table::to_stream(std::ostream &s) const
{
    unsigned int n_cols = m_impl->columns->size();
    if (n_cols)
    {
        s << "\"" << m_impl->columns->get_name(0) << "\"";
        for (unsigned int i = 1; i < n_cols; ++i)
            s << ", \"" << m_impl->columns->get_name(i) << "\"";
        s << std::endl;
    }
    unsigned long long n_rows = this->get_number_of_rows();
    for (unsigned long long j = 0; j < n_rows; ++j)
    {
        if (n_cols)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                m_impl->columns->get(0).get(),
                TT *a = dynamic_cast<TT*>(m_impl->columns->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << v;
                )
            else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
                std::string,
                m_impl->columns->get(0).get(),
                TT *a = dynamic_cast<TT*>(m_impl->columns->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << "\"" << v << "\"";
                )
            for (unsigned int i = 1; i < n_cols; ++i)
            {
                TEMPLATE_DISPATCH(teca_variant_array_impl,
                    m_impl->columns->get(i).get(),
                    TT *a = dynamic_cast<TT*>(m_impl->columns->get(i).get());
                    NT v = NT();
                    a->get(j, v);
                    s << ", " << v;
                    )
                else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
                    std::string,
                    m_impl->columns->get(i).get(),
                    TT *a = dynamic_cast<TT*>(m_impl->columns->get(i).get());
                    NT v = NT();
                    a->get(j, v);
                    s << ", \"" << v << "\"";
                    )
            }
        }
        s << std::endl;
    }
}

// --------------------------------------------------------------------------
void teca_table::copy(const const_p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other.get())
        return;

    this->clear();
    m_impl->metadata = other->m_impl->metadata;
    m_impl->columns->copy(other->m_impl->columns);
}

// --------------------------------------------------------------------------
void teca_table::shallow_copy(const p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    this->clear();
    m_impl->metadata = other->m_impl->metadata;
    m_impl->columns->shallow_copy(other->m_impl->columns);
}

// --------------------------------------------------------------------------
void teca_table::copy_metadata(const const_p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    this->clear();

    m_impl->metadata = other->m_impl->metadata;

    unsigned int n_cols = other->get_number_of_columns();
    for (unsigned int i=0; i<n_cols; ++i)
    {
        m_impl->columns->append(
            other->m_impl->columns->get_name(i),
            other->m_impl->columns->get(i)->new_instance());
    }
}

// --------------------------------------------------------------------------
void teca_table::swap(p_teca_dataset &dataset)
{
    p_teca_table other
        = std::dynamic_pointer_cast<teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    std::shared_ptr<teca_table::impl_t> tmp = m_impl;
    m_impl = other->m_impl;
    other->m_impl = tmp;

    m_impl->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::concatenate(const const_p_teca_table &other)
{
    if (!other)
        return;

    size_t n_cols = 0;
    if ((n_cols = other->get_number_of_columns()) != this->get_number_of_columns())
    {
        TECA_ERROR("append failed. Number of columns don't match")
        return;
    }

    for (size_t i = 0; i < n_cols; ++i)
        this->get_column(i)->append(*(other->get_column(i).get()));
}
