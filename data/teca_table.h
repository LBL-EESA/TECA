#ifndef teca_table_h
#define teca_table_h

#include "teca_dataset.h"
#include "teca_table_fwd.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"

#include <map>
#include <vector>
#include <string>

/**
A collection of collumnar data with row based
accessors and communication and I/O support.
*/
class teca_table : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(teca_table)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_table() = default;

    // set/get metadata
    TECA_DATASET_METADATA(calendar, std::string, 1)
    TECA_DATASET_METADATA(time_units, std::string, 1)

    // remove all column definitions and data
    void clear();

    // define the table columns. requires name,type pairs
    // for ex. define("c1",int(),"c2",float()) creates a
    // table with 2 columns the first storing int's the
    // second storing float's.
    template<typename nT, typename cT, typename... oT>
    void declare_columns(nT &&col_name, cT col_type, oT &&...args);

    // add a column definition to the table.
    template<typename nT, typename cT>
    void declare_column(nT &&col_name, cT col_type);

    // get the number of rows/columns
    unsigned int get_number_of_columns() const noexcept;
    unsigned long get_number_of_rows() const noexcept;

    // get a specific column. return a nullptr if
    // the column doesn't exist.
    p_teca_variant_array get_column(unsigned int i);
    p_teca_variant_array get_column(const std::string &col_name);

    const_p_teca_variant_array get_column(unsigned int i) const;
    const_p_teca_variant_array get_column(const std::string &col_name) const;

    // test for the existance of a specific column
    bool has_column(const std::string &col_name) const
    { return m_impl->columns->has(col_name); }

    // get the name of the column, see also get_number_of_columns
    std::string get_column_name(unsigned int i) const
    { return m_impl->columns->get_name(i); }

    // add a column
    int append_column(p_teca_variant_array array)
    { return m_impl->columns->append(array); }

    int append_column(const std::string &name, p_teca_variant_array array)
    { return m_impl->columns->append(name, array); }

    // remove a column
    int remove_column(unsigned int i)
    { return m_impl->columns->remove(i); }

    int remove_column(const std::string &name)
    { return m_impl->columns->remove(name); }

    // get container holding columns
    p_teca_array_collection get_columns()
    { return m_impl->columns; }

    const_p_teca_array_collection get_columns() const
    { return m_impl->columns; }

    // default initialize n rows of data
    void resize(unsigned long n);

    // reserve memory for future rows
    void reserve(unsigned long n);

    // append the collection of data in succession to
    // each column. see also operator<< for sequential
    // stream insertion like append.
    template<typename cT, typename... oT>
    void append(cT &&val, oT &&... args);

    // covert to bool. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    // return true if the dataset is empty.
    bool empty() const noexcept override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    void to_stream(teca_binary_stream &) const override;
    void from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    void to_stream(std::ostream &) const override;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &other) override;

    // deep copy a subset of row values.
    void copy(const const_p_teca_table &other,
        unsigned long first_row, unsigned long last_row);

    void shallow_copy(const p_teca_dataset &other) override;

    // copy the column layout and types
    void copy_structure(const const_p_teca_table &other);

    // swap internals of the two objects
    void swap(p_teca_dataset &other) override;

    // append rows from the passed in table which must have identical
    // columns.
    void concatenate_rows(const const_p_teca_table &other);

    // append columns from the passed in table which must have same
    // number of rows. if deep flag is true a full copy of the data
    // is made, else a shallow copy is made.
    void concatenate_cols(const const_p_teca_table &other, bool deep=false);

protected:
    teca_table();
    teca_table(const teca_table &other) = default;
    teca_table(teca_table &&other) = default;
    teca_table &operator=(const teca_table &other) = default;
    void declare_columns(){}
    void append(){}

private:
    struct impl_t
    {
        impl_t();
        //
        p_teca_array_collection columns;
        unsigned int active_column;
    };
    std::shared_ptr<impl_t> m_impl;
};





// --------------------------------------------------------------------------
inline
p_teca_variant_array teca_table::get_column(unsigned int i)
{
    return m_impl->columns->get(i);
}

// --------------------------------------------------------------------------
inline
const_p_teca_variant_array teca_table::get_column(unsigned int i) const
{
    return m_impl->columns->get(i);
}

// --------------------------------------------------------------------------
template<typename nT, typename cT, typename... oT>
void teca_table::declare_columns(nT &&col_name, cT col_type, oT &&... args)
{
    m_impl->columns->declare(std::forward<nT>(col_name), col_type);
    this->declare_columns(args...);
}

// --------------------------------------------------------------------------
template<typename nT, typename cT>
void teca_table::declare_column(nT &&col_name, cT col_type)
{
    m_impl->columns->declare(std::forward<nT>(col_name), col_type);
}

// --------------------------------------------------------------------------
template<typename cT, typename... oT>
void teca_table::append(cT &&val, oT &&... args)
{
   unsigned int col = m_impl->active_column++%this->get_number_of_columns();
   m_impl->columns->get(col)->append(std::forward<cT>(val));
   this->append(args...);
}

template<typename T>
p_teca_table &operator<<(p_teca_table &t, T &&v)
{
    t->append(std::forward<T>(v));
    return t;
}

#endif
