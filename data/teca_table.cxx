#include "teca_table.h"

#include "teca_binary_stream.h"

using std::vector;
using std::map;

// --------------------------------------------------------------------------
teca_table::teca_table()
    : impl(teca_array_collection::New()), active_column(0)
{}

// --------------------------------------------------------------------------
p_teca_dataset teca_table::new_instance() const
{
    return p_teca_dataset(new teca_table);
}

// --------------------------------------------------------------------------
p_teca_dataset teca_table::new_copy() const
{
    p_teca_table t(new teca_table(*this));
    return t;
}

// --------------------------------------------------------------------------
void teca_table::clear()
{
    this->impl->clear();
    this->active_column = 0;
}

// --------------------------------------------------------------------------
bool teca_table::empty() const TECA_NOEXCEPT
{
    return this->impl->size() == 0;
}

// --------------------------------------------------------------------------
unsigned int teca_table::get_number_of_columns() const TECA_NOEXCEPT
{
    return this->impl->size();
}

// --------------------------------------------------------------------------
unsigned long teca_table::get_number_of_rows() const TECA_NOEXCEPT
{
    if (this->impl->size())
        return this->impl->get(0)->size();

    return 0;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_table::get_column(const std::string &col_name)
{
    return this->impl->get(col_name);
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_table::get_column(const std::string &col_name) const
{
    return this->impl->get(col_name);
}

// --------------------------------------------------------------------------
void teca_table::resize(unsigned long n)
{
    unsigned int n_cols = this->impl->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->get(i)->resize(n);
}

// --------------------------------------------------------------------------
void teca_table::reserve(unsigned long n)
{
    unsigned int n_cols = this->impl->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->get(i)->reserve(n);
}

// --------------------------------------------------------------------------
void teca_table::to_stream(teca_binary_stream &s) const
{
    this->impl->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_table::from_stream(teca_binary_stream &s)
{
    this->clear();
    this->impl->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_table::to_stream(std::ostream &s) const
{
    unsigned int n_cols = this->impl->size();
    if (n_cols)
    {
        s << this->impl->get_name(0);
        for (unsigned int i = 1; i < n_cols; ++i)
            s << ", " << this->impl->get_name(i);
        s << std::endl;
    }
    unsigned long long n_rows = this->get_number_of_rows();
    for (unsigned long long j = 0; j < n_rows; ++j)
    {
        if (n_cols)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                this->impl->get(0).get(),
                TT *a = dynamic_cast<TT*>(this->impl->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << v;
                )
            else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
                std::string,
                this->impl->get(0).get(),
                TT *a = dynamic_cast<TT*>(this->impl->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << ", \"" << v << "\"";
                )
            for (unsigned int i = 1; i < n_cols; ++i)
            {
                TEMPLATE_DISPATCH(teca_variant_array_impl,
                    this->impl->get(i).get(),
                    TT *a = dynamic_cast<TT*>(this->impl->get(i).get());
                    NT v = NT();
                    a->get(j, v);
                    s << ", " << v;
                    )
                else TEMPLATE_DISPATCH_CASE(teca_variant_array_impl,
                    std::string,
                    this->impl->get(i).get(),
                    TT *a = dynamic_cast<TT*>(this->impl->get(i).get());
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
    this->impl->copy(other->impl);
}

// --------------------------------------------------------------------------
void teca_table::shallow_copy(const p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    this->clear();
    this->impl->shallow_copy(other->impl);
}

// --------------------------------------------------------------------------
void teca_table::copy_metadata(const const_p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    this->clear();

    unsigned int n_cols = other->get_number_of_columns();
    for (unsigned int i=0; i<n_cols; ++i)
    {
        this->impl->append(
            other->impl->get_name(i),
            other->impl->get(i)->new_instance());
    }
}

// --------------------------------------------------------------------------
void teca_table::swap(p_teca_dataset &dataset)
{
    p_teca_table other
        = std::dynamic_pointer_cast<teca_table>(dataset);

    if (!other)
        throw std::bad_cast();

    p_teca_array_collection tmp(this->impl);
    this->impl = other->impl;
    other->impl = tmp;

    this->active_column = 0;
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
