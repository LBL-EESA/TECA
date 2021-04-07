#ifndef teca_cf_time_axis_data_h
#define teca_cf_time_axis_data_h

#include "teca_dataset.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <map>
#include <vector>
#include <string>

#include "teca_shared_object.h"
TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_time_axis_data)

/// A dataset used to read NetCDF CF2 time and metadata in parallel.
class teca_cf_time_axis_data : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(teca_cf_time_axis_data)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    ~teca_cf_time_axis_data() override;

    using elem_t = std::pair<p_teca_variant_array, teca_metadata>;

    // transfer the element associated with file to the dataset.
    // after transfer the passed element is invalid in the calling context
    void transfer(unsigned long file_id, elem_t &&data);

    // access the file's element
    elem_t &get(unsigned long file_id);
    const elem_t &get(unsigned long file_id) const;

    // given an element extract metadata
    static
    teca_metadata &get_metadata(elem_t&elem)
    { return elem.second; }

    static
    const teca_metadata &get_metadata(const elem_t &elem)
    { return elem.second; }

    // given an element extract the time axis
    static
    p_teca_variant_array get_variant_array(elem_t &elem)
    { return elem.first; }

    static
    const_p_teca_variant_array get_variant_array(const elem_t &elem)
    { return elem.first; }

    // append the data from the other instance
    void append(const const_p_teca_dataset &other);
    void shallow_append(const const_p_teca_dataset &other);

    // return a unique string identifier
    std::string get_class_name() const override
    { return "teca_cf_time_axis_data"; }

    // return an integer identifier uniquely naming the dataset type
    int get_type_code() const override;

    // covert to boolean. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    // return true if the dataset is empty.
    bool empty() const noexcept override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    int to_stream(std::ostream &) const override;
    int from_stream(std::istream &) override;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &other) override;

    // deep copy a subset of row values.
    void copy(const const_p_teca_cf_time_axis_data &other,
        unsigned long first_row, unsigned long last_row);

    void shallow_copy(const p_teca_dataset &other) override;

    // swap internals of the two objects
    void swap(p_teca_dataset &other) override;

protected:
    teca_cf_time_axis_data();

    teca_cf_time_axis_data(const teca_cf_time_axis_data &other) = delete;
    teca_cf_time_axis_data(teca_cf_time_axis_data &&other) = delete;
    teca_cf_time_axis_data &operator=(const teca_cf_time_axis_data &other) = delete;

private:
    using internals_t = std::map<unsigned long, elem_t>;
    internals_t *internals;
};

#endif
