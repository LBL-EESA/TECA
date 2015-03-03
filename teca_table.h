#ifndef teca_table_h
#define teca_table_h

#include "teca_dataset.h"
#include "teca_table_fwd.h"
#include "teca_variant_array.h"

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
    virtual ~teca_table() = default;

    TECA_DATASET_COPY_SWAP()

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const override;

    // virtual copy constructor. return a deep copy of this
    // dataset in a new instance.
    virtual p_teca_dataset new_copy() const override;

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
    unsigned int get_number_of_columns() const TECA_NOEXCEPT;
    unsigned long get_number_of_rows() const TECA_NOEXCEPT;

    // get a specific column. return a nullptr if
    // the column doesn't exist.
    p_teca_variant_array get_column(unsigned int i);
    p_teca_variant_array get_column(const std::string &col_name);

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
    explicit operator bool() const TECA_NOEXCEPT
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() const TECA_NOEXCEPT override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) const override;
    virtual void from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    virtual void to_stream(std::ostream &) const override;

protected:
    teca_table();
    teca_table(const teca_table &other) = default;
    teca_table(teca_table &&other) = default;
    teca_table &operator=(teca_table &other) = default;
    void declare_columns(){}
    void append(){}

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    virtual void copy(const teca_dataset *) override;
    virtual void shallow_copy(const teca_dataset *) override;

    // copy metadata. always a deep copy.
    virtual void copy_metadata(const teca_dataset *) override;

    // swap internals of the two objects
    virtual void swap(teca_dataset *) override;

private:
    struct teca_table_impl
    {
        std::vector<std::string> column_names;
        std::vector<p_teca_variant_array> column_data;
        std::map<std::string,unsigned int> name_data_map;
    };
    std::shared_ptr<teca_table_impl> impl;
    unsigned int active_column;
};





// --------------------------------------------------------------------------
inline
p_teca_variant_array teca_table::get_column(unsigned int i)
{
    return this->impl->column_data[i];
}

// --------------------------------------------------------------------------
template<typename nT, typename cT, typename... oT>
void teca_table::declare_columns(nT &&col_name, cT col_type, oT &&... args)
{
    this->declare_column(std::forward<nT>(col_name), col_type);
    //this->declare_columns(std::forward<oT...>(args...));
    this->declare_columns(args...);
}

// --------------------------------------------------------------------------
template<typename nT, typename cT>
void teca_table::declare_column(nT &&col_name, cT)
{
    unsigned int col_id = this->impl->column_data.size();
    this->impl->column_names.emplace_back(std::forward<nT>(col_name));
    this->impl->column_data.emplace_back(teca_variant_array_impl<cT>::New());
    this->impl->name_data_map.emplace(std::forward<nT>(col_name), col_id);
}

// --------------------------------------------------------------------------
template<typename cT, typename... oT>
void teca_table::append(cT &&val, oT &&... args)
{
   unsigned int col = this->active_column++%this->get_number_of_columns();
   this->impl->column_data[col]->append(std::forward<cT>(val));
   //this->append(std::forward<oT...>(args...));
   this->append(args...);
}

template<typename T>
p_teca_table &operator<<(p_teca_table &t, T &&v)
{
    t->append(std::forward<T>(v));
    return t;
}

#endif
