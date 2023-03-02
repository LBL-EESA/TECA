#ifndef teca_table_h
#define teca_table_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"
#include "teca_shared_object.h"

#include <map>
#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table)

/** @brief
 * A collection of columnar data with row based accessors and communication and
 * I/O support.
 */
class TECA_EXPORT teca_table : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(teca_table)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_table() = default;

    /// @name temporal metadata
    /** Specifies the temporal extents of the data and the calendaring system
     * used to define the time axis.
     */
    ///@{
    TECA_DATASET_METADATA(calendar, std::string, 1)
    TECA_DATASET_METADATA(time_units, std::string, 1)
    ///@}

    /// @name attribute metadata
    /** Provides access to the array attributes metadata, which contains
     * information such as dimensions, units, data type, description, etc.
     */
    ///@{
    TECA_DATASET_METADATA(attributes, teca_metadata, 1)
    ///@}

    /// remove all column definitions and data
    void clear();

    /** Define the table columns. requires name,type pairs. For example,
     * declare_columns("c1",int(),"c2",float()) creates a table with 2 columns
     * the first storing int the second storing float.
     */
    template<typename nT, typename cT, typename... oT>
    void declare_columns(nT &&col_name, cT col_type, oT &&...args);

    /// Adds a column definition to the table.
    template<typename nT, typename cT>
    void declare_column(nT &&col_name, cT col_type);

    /// set the allocator to use with ::declare_column
    void set_default_allocator(allocator alloc)
    { m_impl->columns->set_default_allocator(alloc); }

    /// get the number of columns
    unsigned int get_number_of_columns() const noexcept;

    /// get the number of rows
    unsigned long get_number_of_rows() const noexcept;

    /// get a specific column. return a nullptr if the column doesn't exist.
    p_teca_variant_array get_column(unsigned int i);

    /// get a specific column. return a nullptr if the column doesn't exist.
    p_teca_variant_array get_column(const std::string &col_name);

    /// get a specific column. return a nullptr if the column doesn't exist.
    const_p_teca_variant_array get_column(unsigned int i) const;

    /// get a specific column. return a nullptr if the column doesn't exist.
    const_p_teca_variant_array get_column(const std::string &col_name) const;

    /// get a specific column. return a nullptr if the column doesn't exist.
    template <typename array_t>
    std::shared_ptr<array_t> get_column_as(unsigned int i)
    {
        return std::dynamic_pointer_cast<array_t>(this->get_column(i));
    }

    /// get a specific column. return a nullptr if the column doesn't exist.
    template <typename array_t>
    std::shared_ptr<array_t> get_column_as(const std::string &col_name)
    {
        return std::dynamic_pointer_cast<array_t>(this->get_column(col_name));
    }

    /// get a specific column. return a nullptr if the column doesn't exist.
    template <typename array_t>
    std::shared_ptr<const array_t> get_column_as(unsigned int i) const
    {
        return std::dynamic_pointer_cast<const array_t>(this->get_column(i));
    }

    /// get a specific column. return a nullptr if the column doesn't exist.
    template <typename array_t>
    std::shared_ptr<const array_t> get_column_as(const std::string &col_name) const
    {
        return std::dynamic_pointer_cast<const array_t>(this->get_column(col_name));
    }

    /// test for the existence of a specific column
    bool has_column(const std::string &col_name) const
    { return m_impl->columns->has(col_name); }

    /// get the name of column i, see also get_number_of_columns
    std::string get_column_name(unsigned int i) const
    { return m_impl->columns->get_name(i); }

    /// add a column to the table
    int append_column(p_teca_variant_array array)
    { return m_impl->columns->append(array); }

    /// add a column to the table
    int append_column(const std::string &name, p_teca_variant_array array)
    { return m_impl->columns->append(name, array); }

    /// remove a column
    int remove_column(unsigned int i)
    { return m_impl->columns->remove(i); }

    /// remove a column
    int remove_column(const std::string &name)
    { return m_impl->columns->remove(name); }

    /// get the container holding the columns
    p_teca_array_collection get_columns()
    { return m_impl->columns; }

    /// get the container holding the columns
    const_p_teca_array_collection get_columns() const
    { return m_impl->columns; }

    /// resize the table to hold n rows of data, new rows are default initialized
    void resize(unsigned long n);

    /// reserve memory for n rows of data without changing the tables size
    void reserve(unsigned long n);

    /**append the collection of data in succession to each column. see also
     * operator<< for sequential stream insertion like append.
     */
    template<typename cT, typename... oT>
    void append(cT &&val, oT &&... args);

    /// return a unique string identifier
    std::string get_class_name() const override
    { return "teca_table"; }

    /// return an integer identifier uniquely naming the dataset type
    int get_type_code() const override;

    /// covert to boolean. true if the dataset is not empty, otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    /// return true if the dataset is empty.
    bool empty() const noexcept override;

    /// serialize the dataset to the given stream for I/O or communication
    int to_stream(teca_binary_stream &) const override;

    /// deserialize the dataset from the given stream for I/O or communication
    int from_stream(teca_binary_stream &) override;

    /// serialize to the stream in human readable representation
    int to_stream(std::ostream &) const override;

    /// deserialize from the stream in human readable representation
    int from_stream(std::istream &) override;

    /// @copydoc teca_dataset::copy(const const_p_teca_dataset &,allocator)
    void copy(const const_p_teca_dataset &other,
        allocator alloc = allocator::malloc) override;

    /// deep copy a subset of row values.
    void copy(const const_p_teca_table &other,
        unsigned long first_row, unsigned long last_row,
        allocator alloc = allocator::malloc);

    /// @copydoc teca_dataset::shallow_copy(const p_teca_dataset &)
    void shallow_copy(const p_teca_dataset &other) override;

    /// copy the column layout and types
    void copy_structure(const const_p_teca_table &other);

    /// swap internals of the two objects
    void swap(const p_teca_dataset &other) override;

    /// append rows from the passed in table which must have identical columns.
    void concatenate_rows(const const_p_teca_table &other);

    /** append columns from the passed in table which must have same number of
     * rows. if deep flag is true a full copy of the data is made, else a
     * shallow copy is made.
     */
    void concatenate_cols(const const_p_teca_table &other, bool deep=false);

#if defined(SWIG)
protected:
#else
public:
#endif
    // NOTE: constructors are public to enable std::make_shared. do not use.
    teca_table();

protected:
    teca_table(const teca_table &other) = delete;
    teca_table(teca_table &&other) = delete;
    teca_table &operator=(const teca_table &other) = delete;

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
