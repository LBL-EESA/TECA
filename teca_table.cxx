#include "teca_table.h"

#include "teca_binary_stream.h"

using std::vector;
using std::map;

// --------------------------------------------------------------------------
teca_table::teca_table()
    : impl(new teca_table::teca_table_impl), active_column(0)
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
    this->active_column = 0;
    this->impl->column_names.clear();
    this->impl->column_data.clear();
    this->impl->name_data_map.clear();
}

// --------------------------------------------------------------------------
bool teca_table::empty() const TECA_NOEXCEPT
{
    return this->impl->column_data.empty();
}

// --------------------------------------------------------------------------
unsigned int teca_table::get_number_of_columns() const TECA_NOEXCEPT
{
    return this->impl->column_data.size();
}

// --------------------------------------------------------------------------
unsigned long teca_table::get_number_of_rows() const TECA_NOEXCEPT
{
    if (this->impl->column_data.size())
        return this->impl->column_data[0]->size();

    return 0;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_table::get_column(const std::string &col_name)
{
    return this->impl->column_data[this->impl->name_data_map[col_name]];
}

// --------------------------------------------------------------------------
void teca_table::resize(unsigned long n)
{
    unsigned int n_cols = this->impl->column_data.size();
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->column_data[i]->resize(n);
}

// --------------------------------------------------------------------------
void teca_table::reserve(unsigned long n)
{
    unsigned int n_cols = this->impl->column_data.size();
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->column_data[i]->reserve(n);
}

// --------------------------------------------------------------------------
void teca_table::to_stream(teca_binary_stream &s) const
{
    unsigned int n_cols = this->impl->column_data.size();
    s.pack(n_cols);
    s.pack(this->impl->column_names);
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->column_data[i]->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_table::from_stream(teca_binary_stream &s)
{
    unsigned int n_cols;
    s.unpack(n_cols);
    s.unpack(this->impl->column_names);
    for (unsigned int i = 0; i < n_cols; ++i)
        this->impl->column_data[i]->from_stream(s);

    this->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::to_stream(std::ostream &s) const
{
    unsigned int n_cols = this->impl->column_names.size();
    if (n_cols)
    {
        s << this->impl->column_names[0];
        for (unsigned int i = 1; i < n_cols; ++i)
            s << ", " << this->impl->column_names[i];
        s << std::endl;
    }
    unsigned long long n_rows = this->get_number_of_rows();
    for (unsigned long long j = 0; j < n_rows; ++j)
    {
        if (n_cols)
        {
            TEMPLATE_DISPATCH(teca_variant_array_impl,
                this->impl->column_data[0].get(),
                TT *a = dynamic_cast<TT*>(this->impl->column_data[0].get());
                NT v = NT();
                a->get(v, j);
                s << v;
                )
            for (unsigned int i = 1; i < n_cols; ++i)
            {
                TEMPLATE_DISPATCH(teca_variant_array_impl,
                    this->impl->column_data[i].get(),
                    TT *a = dynamic_cast<TT*>(this->impl->column_data[i].get());
                    NT v = NT();
                    a->get(v, j);
                    s << ", " << v;
                    )
            }
        }
        s << std::endl;
    }
}

// --------------------------------------------------------------------------
void teca_table::copy(const teca_dataset *dataset)
{
    const teca_table *other = dynamic_cast<const teca_table*>(dataset);

    if (!other)
        throw std::bad_cast();

    if (this == other)
        return;

    this->impl = std::make_shared<teca_table::teca_table_impl>();

    this->impl->column_names = other->impl->column_names;
    unsigned int n_cols = other->get_number_of_columns();
    this->impl->column_data.resize(n_cols);
    for (unsigned int i=0; i<n_cols; ++i)
        this->impl->column_data[i] = other->impl->column_data[i]->new_copy();
    this->impl->name_data_map = other->impl->name_data_map;

    this->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::shallow_copy(const teca_dataset *dataset)
{
    const teca_table *other = dynamic_cast<const teca_table*>(dataset);

    if (!other)
        throw std::bad_cast();

    this->impl = other->impl;
    this->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::copy_metadata(const teca_dataset *dataset)
{
    const teca_table *other = dynamic_cast<const teca_table*>(dataset);

    if (!other)
        throw std::bad_cast();

    this->impl->column_names = other->impl->column_names;
    unsigned int n_cols = other->get_number_of_columns();
    this->resize(n_cols);
    for (unsigned int i=0; i<n_cols; ++i)
        this->impl->column_data[i] = other->impl->column_data[i]->new_instance();
    this->impl->name_data_map = other->impl->name_data_map;

    this->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::swap(teca_dataset *dataset)
{
    teca_table *other = dynamic_cast<teca_table*>(dataset);

    if (!other)
        throw std::bad_cast();

    std::shared_ptr<teca_table::teca_table_impl> tmp(this->impl);
    this->impl = other->impl;
    other->impl = tmp;

    this->active_column = 0;
}
