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
    TECA_DATASET_METADATA(calendar, std::string, 1, m_impl->metadata)
    TECA_DATASET_METADATA(time_units, std::string, 1, m_impl->metadata)

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

    // get the name of the column, see also get_number_of_columns
    std::string get_column_name(unsigned int i) const
    { return m_impl->columns->get_name(i); }

    // default initialize n rows of data
    void resize(unsigned long n);

    // reserve memory for future rows
    void reserve(unsigned long n);

    // append the collection of data comprising a row. there
    // must be a value provided for each column. see also
    // operator<< for sequential stream like append.
    template<typename cT, typename... oT>
    void append(cT &&val, oT &&... args);

    // covert to bool. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() const noexcept override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) const override;
    virtual void from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    virtual void to_stream(std::ostream &) const override;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    virtual void copy(const const_p_teca_dataset &other) override;
    virtual void shallow_copy(const p_teca_dataset &other) override;

    // copy metadata. always a deep copy.
    virtual void copy_metadata(const const_p_teca_dataset &other) override;

    // swap internals of the two objects
    virtual void swap(p_teca_dataset &other) override;

    // append the passed in table. must have identical
    // column layout.
    void concatenate(const const_p_teca_table &other);

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
        teca_metadata metadata;
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
